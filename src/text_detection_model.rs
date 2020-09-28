use super::dataset::TextDetectionDataset;
use super::image_ops;
use anyhow::Result;
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

fn create_and_train_model() -> Result<FuncT<'static>> {
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
        let test_accuracy = get_model_accuracy(&dataset_paths, &net)?;
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

fn get_model_accuracy(dataset_paths: &TextDetectionDataset, net: &FuncT<'static>) -> Result<f64> {
    // let mut raw_metrics = Vec::new();
    for i in 0..dataset_paths.test_images.len() {
        let images = Tensor::load(&dataset_paths.test_images[i])?;
        let gts = Tensor::load(&dataset_paths.test_gt[i])?;
        let masks = Tensor::load(&dataset_paths.test_mask[i])?;
        let adjs = Tensor::load(&dataset_paths.test_adj[i])?;
        let pred = net.forward_t(&images, false);
        let output = get_boxes_and_box_scores(&images, /* &gts, &masks, */ &adjs, &pred)?;
        // let mut raw_metric = validate_measure(&images, &gts, &masks, &output)?;
        // raw_metrics.append(&mut raw_metric);
    }
    // let metrics = gather_measure(raw_metrics)?;
    Ok(0.)
}

fn get_boxes_and_box_scores(
    images: &Tensor,
    // gts: &Tensor,
    // masks: &Tensor,
    adjust_values: &Tensor,
    pred: &Tensor,
) -> Result<()> {
    let thresh = 0.3;
    let mut boxes_batch = Vec::new();
    let mut scores_batch = Vec::new();
    let segmentation = binarize(&pred, thresh)?;

    for batch_index in 0..images.size()[0] {
        let (boxes, scores) = get_boxes_from_bitmap(
            &pred.get(batch_index),
            &segmentation.get(batch_index),
            &adjust_values.get(batch_index),
        )?;
        boxes_batch.push(boxes);
        scores_batch.push(scores);
    }
    Ok(())
}

fn binarize(pred: &Tensor, thresh: f64) -> Result<Tensor> {
    Ok(pred.gt(thresh).to_kind(Kind::Uint8))
}

fn get_boxes_from_bitmap(
    pred_tensor: &Tensor,
    bitmap_tensor: &Tensor,
    adjust_values: &Tensor,
) -> Result<(Vec<Tensor>, Tensor)> {
    let max_candidates = 1000;
    let min_size = 3.;
    let box_thresh = 0.7;

    let bitmap = bitmap_tensor.get(0); // The first channel
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
        boxes.push(Tensor::new());
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
        boxes[i] = Tensor::of_slice(&box_vec).view((4, 2));
        scores = scores.index_put(
            &[Tensor::of_slice(&[i as i64])],
            &Tensor::of_slice(&[score]),
            true,
        );
    }
    Ok((boxes, scores))
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
    let min_x = clamp(points[0].0.min(points[3].0), 0, w - 1);
    let max_x = clamp(points[1].0.max(points[2].0), 0, w - 1);
    let min_y = clamp(points[0].1.min(points[1].1), 0, h - 1);
    let max_y = clamp(points[2].1.max(points[3].1), 0, h - 1);
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
