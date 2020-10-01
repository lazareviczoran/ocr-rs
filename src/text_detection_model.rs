use super::dataset::TextDetectionDataset;
use super::image_ops;
use anyhow::{anyhow, Result};
use geo::prelude::*;
use geo::{LineString, Polygon};
use geo_booleanop::boolean::BooleanOp;
use image::math::utils::clamp;
use image::{GrayImage, Luma};
use imageproc::drawing::{self, Point};
use log::info;
use tch::{
    nn, nn::Conv2D, nn::ConvTranspose2D, nn::FuncT, nn::ModuleT, nn::OptimizerConfig, Device, Kind,
    Reduction, Tensor,
};

const MODEL_FILENAME: &str = "text_detection_model";
const WHITE_COLOR: Luma<u8> = Luma([255]);

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn conv_transpose2d(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    padding: i64,
    stride: i64,
) -> ConvTranspose2D {
    let conv2d_cfg = nn::ConvTransposeConfig {
        stride,
        padding,
        ..Default::default()
    };
    nn::conv_transpose2d(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    if stride != 1 || c_in != c_out {
        nn::seq_t()
            .add(conv2d(&p / "0", c_in, c_out, 1, 0, stride))
            .add(nn::batch_norm2d(&p / "1", c_out, Default::default()))
    } else {
        nn::seq_t()
    }
}

fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    let conv1 = conv2d(&p / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = nn::batch_norm2d(&p / "bn1", c_out, Default::default());
    let conv2 = conv2d(&p / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = nn::batch_norm2d(&p / "bn2", c_out, Default::default());
    let downsample = downsample(&p / "downsample", c_in, c_out, stride);
    nn::func_t(move |xs, train| {
        let ys = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train);
        (xs.apply_t(&downsample, train) + ys).relu()
    })
}

fn basic_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> impl ModuleT {
    let mut layer = nn::seq_t().add(basic_block(&p / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&p / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}

fn resnet(p: &nn::Path, c1: i64, c2: i64, c3: i64, c4: i64) -> FuncT<'static> {
    let inner_in = 256;
    let conv1 = conv2d(p / "conv1", 1, 64, 7, 3, 2);
    let bn1 = nn::batch_norm2d(p / "bn1", 64, Default::default());
    let layer1 = basic_layer(p / "layer1", 64, 64, 1, c1);
    let layer2 = basic_layer(p / "layer2", 64, 128, 2, c2);
    let layer3 = basic_layer(p / "layer3", 128, 256, 2, c3);
    let layer4 = basic_layer(p / "layer4", 256, 512, 2, c4);

    let in5 = conv2d(p / "in5", 512, inner_in, 1, 0, 1);
    let in4 = conv2d(p / "in4", 256, inner_in, 1, 0, 1);
    let in3 = conv2d(p / "in3", 128, inner_in, 1, 0, 1);
    let in2 = conv2d(p / "in2", 64, inner_in, 1, 0, 1);

    let inner_out = inner_in / 4;
    let out5 = nn::seq()
        .add(conv2d(p / "out5", inner_in, inner_out, 4, 1, 3))
        .add_fn(move |xs| xs.upsample_nearest2d(&[inner_out], 8., 8.));
    let out4 = nn::seq()
        .add(conv2d(p / "out4", inner_in, inner_out, 4, 1, 3))
        .add_fn(move |xs| xs.upsample_nearest2d(&[inner_out], 4., 4.));
    let out3 = nn::seq()
        .add(conv2d(p / "out3", inner_in, inner_out, 4, 1, 3))
        .add_fn(move |xs| xs.upsample_nearest2d(&[inner_out], 2., 2.));
    let out2 = conv2d(p / "out2", inner_in, inner_out, 4, 1, 3);

    // binarization
    let bin_conv1 = conv2d(p / "bin_conv1", inner_in, inner_out, 4, 1, 3);
    let bin_bn1 = nn::batch_norm2d(p / "bin_bn1", inner_out, Default::default());
    let bin_conv_tr1 = conv_transpose2d(p / "bin_conv_tr1", inner_out, inner_out, 2, 0, 2);
    let bin_bn2 = nn::batch_norm2d(p / "bin_bn2", inner_out, Default::default());
    let bin_conv_tr2 = conv_transpose2d(p / "bin_conv_tr2", inner_out, inner_out, 1, 2, 2);

    nn::func_t(move |xs, train| {
        // resnet part
        let x1 = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
            .apply_t(&layer1, train);
        let x2 = x1.apply_t(&layer2, train);
        let x3 = x2.apply_t(&layer3, train);
        let x4 = x3.apply_t(&layer4, train);

        // FPN
        let x_in5 = x4.apply_t(&in5, train);
        let x_in4 = x3.apply_t(&in4, train);
        let x_in3 = x2.apply_t(&in3, train);
        let x_in2 = x1.apply_t(&in2, train);

        let x_out4 = x_in5.upsample_nearest2d(&[inner_in], 2., 2.) + &x_in4;
        let x_out3 = x_in4.upsample_nearest2d(&[inner_in], 2., 2.) + &x_in3;
        let x_out2 = x_in3.upsample_nearest2d(&[inner_in], 2., 2.) + &x_in2;

        let p5 = x_in5.apply_t(&out5, train);
        let p4 = x_out4.apply_t(&out4, train);
        let p3 = x_out3.apply_t(&out3, train);
        let p2 = x_out2.apply_t(&out2, train);

        let fuse = Tensor::cat(&[p5, p4, p3, p2], 1);

        // binarization
        fuse.apply_t(&bin_conv1, train)
            .apply_t(&bin_bn1, train)
            .relu()
            .apply_t(&bin_conv_tr1, train)
            .apply_t(&bin_bn2, train)
            .relu()
            .apply_t(&bin_conv_tr2, train)
            .sigmoid()
    })
}

pub fn resnet18(p: &nn::Path) -> FuncT<'static> {
    resnet(p, 2, 2, 2, 2)
}

pub fn create_and_train_model() -> Result<FuncT<'static>> {
    let dataset_paths = image_ops::load_text_detection_tensor_files("./")?;

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = resnet18(&vs.root());
    let mut opt = nn::sgd(0.9, 0., 0.0001, true).build(&vs, 1e-4)?;
    // for epoch in 1..=1200 {
    for epoch in 1..=1 {
        for i in 0..dataset_paths.train_images.len() {
            let image = Tensor::load(&dataset_paths.train_images[i])?;
            let gt = Tensor::load(&dataset_paths.train_gt[i])?;
            let mask = Tensor::load(&dataset_paths.train_mask[i])?;
            let pred = net.forward_t(&image, true);
            let loss = calculate_balance_loss(pred, gt, mask);
            opt.backward_step(&loss);
        }
        let test_accuracy = get_model_accuracy(&dataset_paths, &net)?.0;
        info!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    vs.save(MODEL_FILENAME)?;

    Ok(net)
}

///
/// Args:
///     pred: shape (N, H, W), the prediction of network
///     gt: shape (N, H, W), the target
///     mask: shape (N, H, W), the mask indicates positive regions
///
fn calculate_balance_loss(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor {
    let negative_ratio = 3.0;
    let eps = 1e-6;

    let positive = &gt * &mask;
    let negative: Tensor = (1 - &gt) * mask;
    let positive_count = positive.sum(Kind::Int64);
    let negative_count = negative
        .sum(Kind::Int64)
        .min1(&(&positive_count * negative_ratio));

    let loss =
        pred.binary_cross_entropy::<Tensor>(&gt.to_kind(Kind::Double), None, Reduction::None);
    let positive_loss = &loss * positive.to_kind(Kind::Double);
    let negative_loss_temp = loss * negative.to_kind(Kind::Double);

    let (negative_loss, _) =
        negative_loss_temp
            .view(-1)
            .topk(negative_count.int64_value(&[]), -1, true, true);

    (positive_loss.sum(Kind::Double) + negative_loss.sum(Kind::Double))
        / (positive_count.to_kind(Kind::Double) + negative_count.to_kind(Kind::Double) + eps)
}

fn get_model_accuracy(
    dataset_paths: &TextDetectionDataset,
    net: &FuncT<'static>,
) -> Result<(f64, f64, f64)> {
    let mut raw_metrics = Vec::new();
    for i in 0..dataset_paths.test_images.len() {
        let images = Tensor::load(&dataset_paths.test_images[i])?;
        let gts = Tensor::load(&dataset_paths.test_gt[i])?;
        let masks = Tensor::load(&dataset_paths.test_mask[i])?;
        let adjs = Tensor::load(&dataset_paths.test_adj[i])?;
        let pred = net.forward_t(&images, false);
        let output = get_boxes_and_box_scores(&pred, &adjs)?;
        // TODO: fix first 2 params (should be polygons + ignore flags)
        raw_metrics.push(validate_measure(&images, &gts, &output.0, &output.1)?);
    }
    let metrics = gather_measure(&raw_metrics)?;
    Ok(metrics)
}

fn get_boxes_and_box_scores(
    pred: &Tensor,
    adjust_values: &Tensor,
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let thresh = 0.3;
    let mut boxes_batch = Vec::new();
    let mut scores_batch = Vec::new();
    let segmentation = binarize(&pred, thresh)?;

    for batch_index in 0..pred.size()[0] {
        let (boxes, scores) = get_boxes_from_bitmap(
            &pred.get(batch_index),
            &segmentation.get(batch_index),
            &adjust_values.get(batch_index),
        )?;
        boxes_batch.push(boxes);
        scores_batch.push(scores);
    }
    Ok((boxes_batch, scores_batch))
}

fn binarize(pred: &Tensor, thresh: f64) -> Result<Tensor> {
    Ok(pred.gt(thresh).to_kind(Kind::Uint8))
}

fn get_boxes_from_bitmap(
    pred_tensor: &Tensor,
    bitmap_tensor: &Tensor,
    adjust_values: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let max_candidates = 1000;
    let min_size = 3.;
    let box_thresh = 0.7;

    let bitmap = bitmap_tensor.get(0);
    let pred = pred_tensor.get(0);
    // convert bitmap to GrayImage
    let image = image_ops::convert_tensor_to_image(&(bitmap * 255))?;

    let contours = image_ops::find_contours(&image)?;
    let num_contours = contours.len().min(max_candidates);
    let mut boxes = Vec::with_capacity(num_contours);
    let mut scores = Tensor::zeros(
        &[num_contours as i64],
        (Kind::Double, Device::cuda_if_available()),
    );

    for i in 0..num_contours {
        let contour = &contours[i];
        boxes.push(Tensor::of_slice(&[-1, -1]).view((-1, 2)));
        let (points, sside) = get_mini_area_bounding_box(contour);
        if sside < min_size {
            continue;
        }
        let score = box_score_fast(&pred, &points)?;
        if box_thresh > score {
            continue;
        }

        let box_vec = points.iter().fold(Vec::new(), |mut acc, p| {
            acc.push((p.0 as f64 / adjust_values.double_value(&[0, 0])).round() as i16);
            acc.push((p.1 as f64 / adjust_values.double_value(&[0, 1])).round() as i16);
            acc
        });
        boxes[i] = Tensor::of_slice(&box_vec).view((-1, 2));
        scores = scores.index_put(
            &[Tensor::of_slice(&[i as i64])],
            &Tensor::of_slice(&[score]),
            true,
        );
    }
    // Merge tensors into one
    let boxes_tensor = merge_point_tensors(&boxes)?;

    Ok((boxes_tensor, scores))
}

fn merge_point_tensors(tensors: &[Tensor]) -> Result<Tensor> {
    if let Some(max_points_num) = tensors.iter().map(|t| t.size()[0]).max() {
        let padded_tensors = tensors
            .iter()
            .map(|t| {
                let diff = max_points_num - t.size()[0];
                if diff > 0 {
                    return Tensor::cat(
                        &[
                            t,
                            &Tensor::of_slice(&vec![-1; diff as usize * 2]).view((-1, 2)),
                        ],
                        0,
                    );
                }
                t.shallow_clone()
            })
            .collect::<Vec<Tensor>>();

        return Ok(Tensor::cat(&padded_tensors, 0).view((tensors.len() as i64, -1, 2)));
    }
    Err(anyhow!("The provided array is empty"))
}

fn get_mini_area_bounding_box(contour: &[(usize, usize)]) -> (Vec<(usize, usize)>, f64) {
    let mut b_box = image_ops::min_area_rect(contour);
    b_box.sort_by(|a, b| a.0.cmp(&b.0));
    let i1 = if b_box[1].1 > b_box[0].1 { 0 } else { 1 };
    let i2 = if b_box[3].1 > b_box[2].1 { 2 } else { 3 };
    let i3 = if b_box[3].1 > b_box[2].1 { 3 } else { 2 };
    let i4 = if b_box[1].1 > b_box[0].1 { 1 } else { 0 };

    let res = vec![b_box[i1], b_box[i2], b_box[i3], b_box[i4]];
    let w = image_ops::get_distance(&res[0], &res[1]);
    let h = image_ops::get_distance(&res[0], &res[3]);
    (res, w.min(h))
}

fn box_score_fast(bitmap: &Tensor, points: &[(usize, usize)]) -> Result<f64> {
    let size = bitmap.size();
    let w = size[size.len() - 2] as usize;
    let h = size[size.len() - 1] as usize;
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        points
            .iter()
            .fold((std::usize::MAX, 0, std::usize::MAX, 0), |acc, p| {
                (
                    acc.0.min(p.0),
                    acc.1.max(p.0),
                    acc.2.min(p.1),
                    acc.3.max(p.1),
                )
            });
    min_x = clamp(min_x, 0, w - 1);
    max_x = clamp(max_x, 0, w - 1);
    min_y = clamp(min_y, 0, h - 1);
    max_y = clamp(max_y, 0, h - 1);
    let mut mask_image = GrayImage::new((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
    let moved_points: Vec<Point<i32>> = points
        .iter()
        .map(|p| Point::new((p.0 - min_x) as i32, (p.1 - min_y) as i32))
        .collect();
    drawing::draw_polygon_mut(&mut mask_image, &moved_points, WHITE_COLOR);

    let mask = (image_ops::convert_image_to_tensor(&mask_image)? / 255).to_kind(Kind::Uint8);

    let partial_bitmap = bitmap
        .narrow(0, min_y as i64, (max_y - min_y + 1) as i64)
        .narrow(1, min_x as i64, (max_x - min_x + 1) as i64)
        * &mask;

    let mean = partial_bitmap.sum(Kind::Double) / mask.sum(Kind::Double);

    Ok(mean.double_value(&[]))
}

//  Args:
//      polygons: tensor of shape (N, K, M, 2), the polygons of objective regions.
//      ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
//      pred: vector of length N of predicted polygons (each tensor of shape (M, 2) coresponds to a single image)
//      scores: vector of length N of predicted polygons scores
fn validate_measure(
    polygons: &Tensor,
    ignore_tags: &Tensor,
    pred: &[Tensor],
    scores: &[Tensor],
    // is_output_polygon: bool,
) -> Result<Vec<MetricsItem>> {
    let is_output_polygon = false;
    let box_thresh = 0.6;
    let mut results = Vec::new();
    for i in 0..polygons.size()[0] {
        let curr_polygons = polygons.get(i);
        let curr_ignore_tags = ignore_tags.get(i);

        let mut pred_points = Vec::new();
        for j in 0..pred[i as usize].size()[0] {
            if is_output_polygon || scores[i as usize].get(j).double_value(&[]) >= box_thresh {
                pred_points.push(pred[i as usize].get(j));
            }
        }

        results.push(evaluate_image(
            (&curr_polygons, &curr_ignore_tags),
            &pred_points,
        )?);
    }
    Ok(results)
}

fn gather_measure(metrics: &[Vec<MetricsItem>]) -> Result<(f64, f64, f64)> {
    let raw_metrics = metrics.iter().fold(Vec::new(), |mut acc, batch_metrics| {
        acc.append(&mut batch_metrics.to_vec());
        acc
    });

    Ok(combine_results(&raw_metrics)?)
}

fn combine_results(results: &[MetricsItem]) -> Result<(f64, f64, f64)> {
    let mut num_global_care_gt = 0;
    let mut num_global_care_det = 0;
    let mut matched_sum = 0;

    for res in results {
        num_global_care_gt += res.gt_care;
        num_global_care_det += res.det_care;
        matched_sum += res.det_matched;
    }
    let method_recall = if num_global_care_gt == 0 {
        0.
    } else {
        matched_sum as f64 / num_global_care_gt as f64
    };
    let method_precision = if num_global_care_det == 0 {
        0.
    } else {
        matched_sum as f64 / num_global_care_det as f64
    };
    let method_hmean = if method_recall + method_precision == 0. {
        0.
    } else {
        2. * (method_recall * method_precision) / (method_recall + method_precision)
    };
    Ok((method_precision, method_recall, method_hmean))
}

fn evaluate_image(gt: (&Tensor, &Tensor), pred: &[Tensor]) -> Result<MetricsItem> {
    let area_precision_constraint = 0.5;
    let iou_constraint = 0.5;

    let mut det_matched = 0;

    let mut gt_polys = Vec::new();
    let mut det_polys = Vec::new();

    let mut gt_dont_care_pols_num = Vec::new();
    let mut det_dont_care_pols_num = Vec::new();

    let mut pairs = Vec::new();
    let mut det_matched_nums = Vec::new();

    for n in 0..gt.0.size()[0] {
        let mut points = Vec::new();
        let gt_points = gt.0.get(n);
        for i in 0..gt_points.size()[0] {
            let x = gt_points.double_value(&[i, 0]);
            let y = gt_points.double_value(&[i, 1]);
            if x < 0. {
                break;
            }
            points.push((x, y));
        }
        let polygon = Polygon::new(LineString::from(points), vec![]);
        let dont_care = gt.1.get(n).double_value(&[]) > 0.;

        gt_polys.push(polygon);
        if dont_care {
            gt_dont_care_pols_num.push(gt_polys.len() - 1);
        }
    }

    for pred_tensor in pred {
        let mut points = Vec::new();
        for i in 0..pred_tensor.size()[0] {
            let x = pred_tensor.double_value(&[i, 0]);
            let y = pred_tensor.double_value(&[i, 1]);
            if x < 0. {
                break;
            }
            points.push((x, y));
        }
        let polygon = Polygon::new(LineString::from(points), vec![]);
        det_polys.push(polygon.clone());
        if !gt_dont_care_pols_num.is_empty() {
            for &dont_care_poly in &gt_dont_care_pols_num {
                let intersected_area = get_intersection(&gt_polys[dont_care_poly], &polygon)?;
                let pd_area = polygon.unsigned_area();
                let precision = if pd_area == 0. {
                    0.
                } else {
                    intersected_area / pd_area
                };
                if precision > area_precision_constraint {
                    det_dont_care_pols_num.push(det_polys.len() - 1);
                    break;
                }
            }
        }
    }

    if !gt_polys.is_empty() && !det_polys.is_empty() {
        let size = [gt_polys.len(), det_polys.len()];
        let mut iou_mat = vec![vec![0.; size[1]]; size[0]];
        let mut gt_rect_mat = vec![0; gt_polys.len()];
        let mut det_rect_mat = vec![0; det_polys.len()];
        for gt_num in 0..gt_polys.len() {
            for det_num in 0..det_polys.len() {
                iou_mat[gt_num][det_num] =
                    get_intersection_over_union(&det_polys[det_num], &gt_polys[gt_num])?;

                if gt_rect_mat[gt_num] == 0
                    && det_rect_mat[det_num] == 0
                    && !gt_dont_care_pols_num.contains(&gt_num)
                    && !gt_dont_care_pols_num.contains(&det_num)
                    && iou_mat[gt_num][det_num] > iou_constraint
                {
                    gt_rect_mat[gt_num] = 1;
                    det_rect_mat[det_num] = 1;
                    det_matched += 1;
                    pairs.push((gt_num, det_num));
                    det_matched_nums.push(det_num);
                }
            }
        }
    }

    let num_gt_care = gt_polys.len() - gt_dont_care_pols_num.len();
    let num_det_care = det_polys.len() - det_dont_care_pols_num.len();
    let recall;
    let precision;
    if num_gt_care == 0 {
        recall = 1.;
        precision = if num_det_care > 0 { 0. } else { 1. };
    } else {
        recall = det_matched as f64 / num_gt_care as f64;
        precision = if num_det_care == 0 {
            0.
        } else {
            det_matched as f64 / num_det_care as f64
        };
    }

    let hmean;
    if precision + recall == 0. {
        hmean = 0.;
    } else {
        hmean = 2. * precision * recall / (precision + recall);
    };

    Ok(MetricsItem {
        precision,
        recall,
        hmean,
        gt_care: num_gt_care,
        det_care: num_det_care,
        det_matched,
    })
}

fn get_intersection(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    let intersection = poly1.intersection(poly2);
    Ok(intersection.unsigned_area())
}

fn get_union(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    let union = poly1.union(poly2);
    Ok(union.unsigned_area())
}

fn get_intersection_over_union(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    Ok(get_intersection(poly1, poly2)? / get_union(poly1, poly2)?)
}

#[derive(Copy, Clone, Debug)]
struct MetricsItem {
    precision: f64,
    recall: f64,
    hmean: f64,
    gt_care: usize,
    det_care: usize,
    det_matched: usize,
}

#[cfg(test)]
mod tests {
    use super::image_ops::convert_image_to_tensor;
    use super::*;
    use image::open;
    use imageproc::drawing::draw_polygon_mut;
    const ERROR_THRESHOLD: f64 = 1e-10;

    #[test]
    fn get_mini_area_bounding_box_test() {
        let res = get_mini_area_bounding_box(&[(141, 24), (61, 16), (57, 53), (137, 61)]);
        assert_eq!(res.0, vec![(60, 15), (142, 23), (138, 62), (57, 54)]);
        assert!(res.1 - 39.11521443121589 < ERROR_THRESHOLD);
    }

    #[test]
    fn box_score_with_whole_matrix_test() {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![(0, 0), (4, 0), (4, 4), (0, 4)];
        assert_eq!(box_score_fast(&pred, &points).unwrap(), 10. / 25.);
    }

    #[test]
    fn box_score_partial_matrix_test() {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![(1, 0), (4, 0), (4, 3), (1, 3)];
        assert_eq!(box_score_fast(&pred, &points).unwrap(), 8. / 16.);
    }

    #[test]
    fn box_score_partial_matrix_test_2() {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![(2, 0), (4, 1), (2, 4), (1, 3)];
        assert_eq!(box_score_fast(&pred, &points).unwrap(), 9. / 12.);
    }

    #[test]
    fn binarize_test() -> Result<()> {
        let values = vec![
            0.01, 0.2, 0.57, 0.58, 0.18, //
            0.39, 0.01, 0.61, 1.0, 0.42, //
            0.4, 0.94, 0.835, 0.793, 0.32, //
            0.57, 0.77, 0.62, 0.51, 0.29, //
            0.11, 0.69, 0.59, 0.21, 0.35, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        assert_eq!(
            binarize(&pred, 0.57)?,
            Tensor::of_slice(&[
                0, 0, 0, 1, 0, //
                0, 0, 1, 1, 0, //
                0, 1, 1, 1, 0, //
                0, 1, 1, 0, 0, //
                0, 1, 1, 0, 0, //
            ])
            .view((5, 5))
        );
        Ok(())
    }

    #[test]
    fn get_boxes_from_bitmap_test_perfect_match_with_2x_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        // perfect match
        let mut prediction_image = GrayImage::new(w as u32, h as u32);
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(60, 16),
                Point::new(141, 24),
                Point::new(137, 61),
                Point::new(57, 53),
            ],
            WHITE_COLOR,
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(235, 17),
                Point::new(302, 21),
                Point::new(299, 59),
                Point::new(233, 54),
            ],
            WHITE_COLOR,
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(7, 75),
                Point::new(148, 95),
                Point::new(145, 111),
                Point::new(4, 91),
            ],
            WHITE_COLOR,
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(250, 79),
                Point::new(303, 85),
                Point::new(299, 126),
                Point::new(245, 120),
            ],
            WHITE_COLOR,
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(25, 181),
                Point::new(125, 181),
                Point::new(125, 268),
                Point::new(25, 268),
            ],
            WHITE_COLOR,
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(190, 250),
                Point::new(255, 185),
                Point::new(285, 215),
                Point::new(220, 280),
            ],
            WHITE_COLOR,
        );
        let pred_tensor = (convert_image_to_tensor(&prediction_image)? / 255.).view((1, h, w));

        // 2x adjustment
        let adjust_values = Tensor::of_slice(&[2., 2.]).view((1, 2));
        let expected_boxes = Tensor::of_slice(&[
            // expected boxes are reajdusted to the original image size (divided by adjust values)
            30i16, 8, 71, 12, 69, 31, 29, 27, // 1
            118i16, 9, 151, 11, 150, 30, 117, 27, // 2
            4i16, 38, 74, 48, 73, 56, 2, 46, // 3
            125i16, 40, 152, 43, 150, 63, 123, 60, // 4
            13i16, 91, 63, 91, 63, 134, 13, 134, // 5
            95i16, 125, 128, 93, 143, 108, 110, 140, // 6
        ])
        .view((6, -1, 2));
        let expected_scores = Tensor::ones(&[6], (Kind::Double, Device::cuda_if_available()));
        assert_eq!(
            get_boxes_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_boxes_from_bitmap_test_partial_match_with_no_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // no adjustment
        let adjust_values = Tensor::of_slice(&[1., 1.]).view((1, 2));
        let expected_boxes = Tensor::of_slice(&[
            // expected boxes with score over thresh (0.7) should appear
            -1, -1, -1, -1, -1, -1, -1, -1, // 1
            -1, -1, -1, -1, -1, -1, -1, -1, // 2
            -1, -1, -1, -1, -1, -1, -1, -1, // 3
            250, 79, 303, 85, 299, 126, 245, 120, // 4
            25, 181, 125, 181, 125, 268, 25, 268, // 5
            -1, -1, -1, -1, -1, -1, -1, -1, // 6
        ])
        .view((6, -1, 2));
        let expected_scores =
            Tensor::of_slice(&[0., 0., 0., 0.703405017921147, 0.7521377137713772, 0.]);
        assert_eq!(
            get_boxes_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn merge_point_tensors_test() -> Result<()> {
        let tensors = [
            Tensor::of_slice(&[10, 10, 20, 10, 20, 20, 10, 20]).view((-1, 2)),
            Tensor::of_slice(&[100, 10, 120, 10, 140, 20, 120, 20, 100, 20]).view((-1, 2)),
            Tensor::of_slice(&[
                100, 100, 120, 100, 140, 120, 120, 140, 100, 140, 80, 120, 90, 110,
            ])
            .view((-1, 2)),
            Tensor::of_slice(&[10, 100, 20, 100, 40, 120, 20, 120]).view((-1, 2)),
        ];
        assert_eq!(
            merge_point_tensors(&tensors)?,
            Tensor::of_slice(&[
                10, 10, 20, 10, 20, 20, 10, 20, -1, -1, -1, -1, -1, -1, // poly 1
                100, 10, 120, 10, 140, 20, 120, 20, 100, 20, -1, -1, -1, -1, // poly 2
                100, 100, 120, 100, 140, 120, 120, 140, 100, 140, 80, 120, 90, 110, // poly 3
                10, 100, 20, 100, 40, 120, 20, 120, -1, -1, -1, -1, -1, -1 // poly 4
            ])
            .view((4, -1, 2))
        );
        Ok(())
    }

    #[test]
    fn evaluate_image_test_one_matching_polygon() -> Result<()> {
        let gt_polygons_tensor = Tensor::of_slice(&[
            0, 0, 10, 0, 10, 10, 0, 10, // poly 1
            20, 20, 30, 20, 30, 30, 20, 30, // poly 2
        ])
        .view((2, -1, 2));
        let ignore_polygons_tensor = Tensor::of_slice(&[0, 0]);
        let pred = [Tensor::of_slice(&[1, 1, 10, 0, 10, 10, 0, 10]).view((-1, 2))];
        let metrics = evaluate_image((&gt_polygons_tensor, &ignore_polygons_tensor), &pred)?;

        assert_eq!(metrics.gt_care, 2);
        assert_eq!(metrics.det_care, 1);
        assert_eq!(metrics.det_matched, 1);
        assert!(metrics.precision - 1. < ERROR_THRESHOLD);
        assert!(metrics.recall - 0.5 < ERROR_THRESHOLD);
        assert!(metrics.hmean - 0.6666666666666666 < ERROR_THRESHOLD);

        Ok(())
    }

    #[test]
    fn evaluate_image_test_with_ignored_polygons() -> Result<()> {
        let gt_polygons_tensor = Tensor::of_slice(&[
            0, 0, 10, 0, 10, 10, 0, 10, // poly 1
            20, 20, 30, 20, 30, 30, 20, 30, // poly 2
        ])
        .view((2, -1, 2));
        let ignore_polygons_tensor = Tensor::of_slice(&[1, 1]);
        let pred = [Tensor::of_slice(&[1, 1, 10, 0, 10, 10, 0, 10]).view((-1, 2))];
        let metrics = evaluate_image((&gt_polygons_tensor, &ignore_polygons_tensor), &pred)?;

        assert_eq!(metrics.gt_care, 0);
        assert_eq!(metrics.det_care, 0);
        assert_eq!(metrics.det_matched, 0);
        assert!(metrics.precision - 1. < ERROR_THRESHOLD);
        assert!(metrics.recall - 1. < ERROR_THRESHOLD);
        assert!(metrics.hmean - 1. < ERROR_THRESHOLD);

        Ok(())
    }

    #[test]
    fn evaluate_image_test_with_both_matched_polygons() -> Result<()> {
        let gt_polygons_tensor = Tensor::of_slice(&[
            0, 0, 10, 0, 10, 10, 0, 10, // poly 1
            20, 20, 30, 20, 30, 30, 20, 30, // poly 2
        ])
        .view((2, -1, 2));
        let ignore_polygons_tensor = Tensor::of_slice(&[0, 0]);
        let pred = [
            Tensor::of_slice(&[1, 1, 10, 0, 10, 10, 0, 10]).view((-1, 2)),
            Tensor::of_slice(&[20, 20, 30, 20, 30, 30, 20, 30]).view((-1, 2)),
        ];
        let metrics = evaluate_image((&gt_polygons_tensor, &ignore_polygons_tensor), &pred)?;

        assert_eq!(metrics.gt_care, 2);
        assert_eq!(metrics.det_care, 2);
        assert_eq!(metrics.det_matched, 2);
        assert!(metrics.precision - 1. < ERROR_THRESHOLD);
        assert!(metrics.recall - 1. < ERROR_THRESHOLD);
        assert!(metrics.hmean - 1. < ERROR_THRESHOLD);

        Ok(())
    }

    #[test]
    fn validate_measure_test() -> Result<()> {
        let gt_polygons_tensor = Tensor::of_slice(&[
            // image 1
            0, 0, 10, 0, 10, 10, 0, 10, // poly 1
            20, 20, 30, 20, 30, 30, 20, 30, // poly 2
            // image 2
            0, 0, 10, 0, 10, 10, 0, 10, // poly 1
            20, 20, 30, 20, 30, 30, 20, 30, // poly 2
        ])
        .view((2, 2, -1, 2));
        let ignore_polygons_tensor = Tensor::of_slice(&[
            0, 0, // image 1
            0, 0, // image 2
        ])
        .view((2, -1));
        let pred = [
            Tensor::of_slice(&[1, 1, 10, 0, 10, 10, 0, 10]).view((1, -1, 2)),
            Tensor::of_slice(&[45, 61, 47, 41, 60, 60, 39, 48]).view((1, -1, 2)),
        ];
        let scores = [Tensor::of_slice(&[0.9]), Tensor::of_slice(&[0.9])];
        let metrics =
            validate_measure(&gt_polygons_tensor, &ignore_polygons_tensor, &pred, &scores)?;
        assert_eq!(metrics.len(), 2);

        // image 1 metrics (having 1 matched polygon)
        assert_eq!(metrics[0].gt_care, 2);
        assert_eq!(metrics[0].det_care, 1);
        assert_eq!(metrics[0].det_matched, 1);
        assert!(metrics[0].precision - 1. < ERROR_THRESHOLD);
        assert!(metrics[0].recall - 0.5 < ERROR_THRESHOLD);
        assert!(metrics[0].hmean - 0.6666666666666666 < ERROR_THRESHOLD);

        // image 2 metrics (having 0 matched polygons)
        assert_eq!(metrics[1].gt_care, 2);
        assert_eq!(metrics[1].det_care, 1);
        assert_eq!(metrics[1].det_matched, 0);
        assert!(metrics[1].precision < ERROR_THRESHOLD);
        assert!(metrics[1].recall < ERROR_THRESHOLD);
        assert!(metrics[1].hmean < ERROR_THRESHOLD);
        Ok(())
    }

    #[test]
    fn combine_results_test() -> Result<()> {
        let metrics = [
            MetricsItem {
                precision: 1.,
                recall: 0.5,
                hmean: 0.6666666666666666,
                gt_care: 2,
                det_care: 1,
                det_matched: 1,
            },
            MetricsItem {
                precision: 1.,
                recall: 1.,
                hmean: 1.,
                gt_care: 0,
                det_care: 0,
                det_matched: 0,
            },
            MetricsItem {
                precision: 1.,
                recall: 1.,
                hmean: 1.,
                gt_care: 2,
                det_care: 2,
                det_matched: 2,
            },
            MetricsItem {
                precision: 0.3333333333333333,
                recall: 0.2,
                hmean: 0.25,
                gt_care: 5,
                det_care: 3,
                det_matched: 1,
            },
        ];
        assert_eq!(
            combine_results(&metrics)?,
            (0.6666666666666666, 0.4444444444444444, 0.5333333333333333)
        );
        Ok(())
    }

    #[test]
    fn gather_measure_test() -> Result<()> {
        let metrics = [
            vec![
                MetricsItem {
                    precision: 1.,
                    recall: 0.5,
                    hmean: 0.6666666666666666,
                    gt_care: 2,
                    det_care: 1,
                    det_matched: 1,
                },
                MetricsItem {
                    precision: 1.,
                    recall: 1.,
                    hmean: 1.,
                    gt_care: 0,
                    det_care: 0,
                    det_matched: 0,
                },
            ],
            vec![
                MetricsItem {
                    precision: 1.,
                    recall: 1.,
                    hmean: 1.,
                    gt_care: 2,
                    det_care: 2,
                    det_matched: 2,
                },
                MetricsItem {
                    precision: 0.3333333333333333,
                    recall: 0.2,
                    hmean: 0.25,
                    gt_care: 5,
                    det_care: 3,
                    det_matched: 1,
                },
            ],
        ];
        assert_eq!(
            gather_measure(&metrics)?,
            (0.6666666666666666, 0.4444444444444444, 0.5333333333333333)
        );
        Ok(())
    }
}
