use super::dataset::TextDetectionDataset;
use super::image_ops::{self, BatchPolygons, MultiplePolygons};
use super::DEVICE;
use crate::utils::{parse_dimensions, parse_number, save_vs};
use anyhow::{anyhow, Result};
use geo::prelude::*;
use geo::{LineString, Polygon};
use geo_booleanop::boolean::BooleanOp;
use image::{GrayImage, Luma};
use imageproc::contours::{approx_poly_dp, arc_length, find_contours, get_distance, min_area_rect};
use imageproc::definitions::{HasWhite, Point};
use imageproc::drawing::{self};
use log::{debug, info};
use num_traits::clamp;
use std::mem::drop;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tch::{
    nn, nn::Conv2D, nn::ConvTranspose2D, nn::FuncT, nn::ModuleT, nn::OptimizerConfig, Kind,
    Reduction, Tensor,
};

pub const MODEL_FILENAME: &str = "text_detection.model";
const DEFAULT_TENSORS_DIR: &str = "./text_det_tensor_files";
pub const DEFAULT_WIDTH: u32 = 800;
pub const DEFAULT_HEIGHT: u32 = 800;
lazy_static! {
    static ref COUNTER: Mutex<[usize; 1]> = Mutex::new([0]);
}

#[derive(Debug)]
pub struct TextDetOptions<'a> {
    epoch: usize,
    model_file_path: &'a str,
    tensor_files_dir: &'a str,
    learning_rate: f64,
    momentum: f64,
    dampening: f64,
    weight_decay: f64,
    nesterov: bool,
    image_dimensions: (u32, u32),
    test_interval: usize,
    resume: bool,
}

impl Default for TextDetOptions<'_> {
    fn default() -> Self {
        Self {
            epoch: 1200,
            model_file_path: MODEL_FILENAME,
            tensor_files_dir: DEFAULT_TENSORS_DIR,
            learning_rate: 1e-4,
            momentum: 0.9,
            dampening: 0.,
            weight_decay: 1e-4,
            nesterov: true,
            image_dimensions: (DEFAULT_WIDTH, DEFAULT_HEIGHT),
            test_interval: 200,
            resume: false,
        }
    }
}
impl<'a> TextDetOptions<'a> {
    pub fn new(args: &'a clap::ArgMatches) -> Result<Self> {
        let mut opts = Self::default();
        if let Some(epoch_str) = args.value_of("epoch") {
            opts.epoch = parse_number(epoch_str, "epoch")?;
        }
        if let Some(path) = args.value_of("model-file") {
            opts.model_file_path = path;
        }
        if let Some(path) = args.value_of("tensor-files-dir") {
            opts.tensor_files_dir = path;
        }
        if let Some(lr_str) = args.value_of("learning-rate") {
            opts.learning_rate = parse_number(lr_str, "learning rate")?;
        }
        if let Some(momentum_str) = args.value_of("momentum") {
            opts.momentum = parse_number(momentum_str, "momentum")?;
        }
        if let Some(dampening_str) = args.value_of("dampening") {
            opts.dampening = parse_number(dampening_str, "dampening")?;
        }
        if let Some(wd_str) = args.value_of("weight-decay") {
            opts.weight_decay = parse_number(wd_str, "weight decay")?;
        }
        if args.is_present("no-nesterov") {
            opts.nesterov = false;
        }
        if let Some(dims_str) = args.value_of("dimensions") {
            opts.image_dimensions = parse_dimensions(dims_str)?;
        }
        if let Some(interval_str) = args.value_of("test-interval") {
            opts.test_interval = parse_number(interval_str, "interval")?;
        }
        if args.is_present("resume") {
            opts.resume = true;
        }

        Ok(opts)
    }
}

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
        (xs.apply_t(&conv1, train)
            .apply_t(&bn1, train)
            .relu()
            .apply_t(&conv2, train)
            .apply_t(&bn2, train)
            + xs.apply_t(&downsample, train))
        .relu()
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
    let inner_out = inner_in / 4;
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

    let out5 = nn::seq()
        .add(conv2d(p / "out5", inner_in, inner_out, 3, 1, 1))
        .add_fn(move |xs| {
            let size = xs.size();
            xs.upsample_nearest2d(&[size[2] * 8, size[3] * 8], None, None)
        });
    let out4 = nn::seq()
        .add(conv2d(p / "out4", inner_in, inner_out, 3, 1, 1))
        .add_fn(move |xs| {
            let size = xs.size();
            xs.upsample_nearest2d(&[size[2] * 4, size[3] * 4], None, None)
        });
    let out3 = nn::seq()
        .add(conv2d(p / "out3", inner_in, inner_out, 3, 1, 1))
        .add_fn(move |xs| {
            let size = xs.size();
            xs.upsample_nearest2d(&[size[2] * 2, size[3] * 2], None, None)
        });
    let out2 = conv2d(p / "out2", inner_in, inner_out, 3, 1, 1);

    // binarization
    let bin_conv1 = conv2d(p / "bin_conv1", inner_in, inner_out, 3, 1, 1);
    let bin_bn1 = nn::batch_norm2d(p / "bin_bn1", inner_out, Default::default());
    let bin_conv_tr1 = conv_transpose2d(p / "bin_conv_tr1", inner_out, inner_out, 2, 0, 2);
    let bin_bn2 = nn::batch_norm2d(p / "bin_bn2", inner_out, Default::default());
    let bin_conv_tr2 = conv_transpose2d(p / "bin_conv_tr2", inner_out, 1, 2, 0, 2);

    nn::func_t(move |xs, train| {
        let x1 = xs
            .apply_t(&conv1, train)
            .apply_t(&bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
            .apply_t(&layer1, train);
        let x2 = x1.apply_t(&layer2, train);
        let x_in2 = x1.apply_t(&in2, train);
        drop(x1);
        let x3 = x2.apply_t(&layer3, train);
        let x_in3 = x2.apply_t(&in3, train);
        drop(x2);
        let x4 = x3.apply_t(&layer4, train);
        let x_in4 = &x3.apply_t(&in4, train);
        drop(x3);
        let x_in5 = x4.apply_t(&in5, train);
        drop(x4);

        let x_in3_size = x_in3.size();
        let p2 = (x_in3.upsample_nearest2d(&[x_in3_size[2] * 2, x_in3_size[3] * 2], None, None)
            + x_in2)
            .apply_t(&out2, train);
        let x_in4_size = x_in4.size();
        let p3 = (x_in4.upsample_nearest2d(&[x_in4_size[2] * 2, x_in4_size[3] * 2], None, None)
            + x_in3)
            .apply_t(&out3, train);
        let x_in5_size = x_in5.size();
        let p4 = (x_in5.upsample_nearest2d(&[x_in5_size[2] * 2, x_in5_size[3] * 2], None, None)
            + x_in4)
            .apply_t(&out4, train);
        let p5 = x_in5.apply_t(&out5, train);

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

pub fn run_text_detection(
    image_path: &str,
    model_file_path: &str,
    dimensions: (u32, u32),
) -> Result<()> {
    let mut instant = Instant::now();
    let (preprocessed_image, adj_x, adj_y) = image_ops::preprocess_image(image_path, dimensions)?;
    debug!("preprocessed in {} ms", instant.elapsed().as_millis());
    instant = Instant::now();
    let mut vs = nn::VarStore::new(*DEVICE);
    if !Path::new(model_file_path).exists() {
        return Err(anyhow!("Model file doesn't exist"));
    }
    let net = resnet18(&vs.root());
    vs.load(model_file_path)?;
    debug!("loaded model in {} ms", instant.elapsed().as_millis());
    instant = Instant::now();
    let image_tensor =
        image_ops::convert_image_to_tensor(&preprocessed_image)?.to_kind(Kind::Float);
    debug!("image -> tensor in {} ms", instant.elapsed().as_millis());

    instant = Instant::now();
    let (w, h) = dimensions;
    let pred = net.forward_t(&image_tensor.view((1, 1, h as i64, w as i64)), false);
    debug!("inference in {} ms", instant.elapsed().as_millis());

    instant = Instant::now();
    let pred_image = image_ops::convert_tensor_to_image(&pred.get(0).get(0))?;
    debug!("tensor -> image in {} ms", instant.elapsed().as_millis());
    instant = Instant::now();
    pred_image.save("pred.png")?;
    debug!("saved image in {} ms", instant.elapsed().as_millis());
    instant = Instant::now();
    let (_boxes, _scores) =
        get_boxes_and_box_scores(&pred, &Tensor::of_slice(&[adj_x, adj_y]).view((1, 2)), true)?;
    debug!("found contours in {} ms", instant.elapsed().as_millis());
    Ok(())
}

pub fn train_model(opts: &TextDetOptions) -> Result<FuncT<'static>> {
    let epoch_limit = opts.epoch;
    let dataset_paths = image_ops::load_text_detection_tensor_files(opts.tensor_files_dir)?;

    let mut vs = nn::VarStore::new(*DEVICE);
    let net = resnet18(&vs.root());
    if opts.resume {
        if !Path::new(opts.model_file_path).exists() {
            return Err(anyhow!("File {} doesn't exist", opts.model_file_path));
        }
        vs.load(opts.model_file_path)?;
    } else if Path::new(opts.model_file_path).exists() {
        return Err(anyhow!(
            "File {} already exist (use '--resume' flag to continue training on existing model, or rename/remove existing file",
            opts.model_file_path
        ));
    }
    let mut opt = nn::sgd(
        opts.momentum,
        opts.dampening,
        opts.weight_decay,
        opts.nesterov,
    )
    .build(&vs, opts.learning_rate)?;
    let (w, h) = (
        opts.image_dimensions.0 as i64,
        opts.image_dimensions.1 as i64,
    );
    for epoch in 1..=epoch_limit {
        let instant = Instant::now();
        for (i, images_batch) in dataset_paths.train_images.iter().enumerate() {
            let load_instant = Instant::now();

            debug!("data batch: {}", i);
            let image = Tensor::load(&images_batch)?
                .view((-1, 1, h, w))
                .to_kind(Kind::Float);
            let gt = Tensor::load(&dataset_paths.train_gt[i])?;
            let mask = Tensor::load(&dataset_paths.train_mask[i])?;
            debug!(
                "loaded single batch in {} ms",
                load_instant.elapsed().as_millis(),
            );
            for j in 0..image.size()[0] {
                let mut train_instant = Instant::now();
                let pred = net.forward_t(&image.get(j).view((1, 1, h, w)), true);
                let train_pred_time = train_instant.elapsed().as_millis();
                train_instant = Instant::now();
                let loss = calculate_balance_loss(pred, gt.get(j), mask.get(j));
                let loss_time = train_instant.elapsed().as_millis();
                train_instant = Instant::now();
                opt.backward_step(&loss);
                let backward_time = train_instant.elapsed().as_millis();
                debug!(
                        "completed single training prediction in {} ms (pred {} ms, loss {} ms, backward {} ms",
                        train_pred_time + loss_time + backward_time,
                        train_pred_time, loss_time, backward_time
                    );
            }
            save_vs(&vs, opts.model_file_path)?;
        }
        debug!(
            "Finished training single batch in {} ms",
            instant.elapsed().as_millis()
        );
        if epoch % opts.test_interval == 0 || epoch == epoch_limit {
            let test_instant = Instant::now();
            let (precision, _recall, _hmean) =
                get_model_accuracy(&dataset_paths, &net, true, opts.image_dimensions)?;
            debug!(
                "Finished test process in {} ms",
                test_instant.elapsed().as_millis()
            );
            info!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * precision);
        }
    }

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
    let positive_count = positive.sum(Kind::Float);
    let negative_count = negative
        .sum(Kind::Float)
        .min1(&(&positive_count * negative_ratio));

    let loss = pred.binary_cross_entropy::<Tensor>(&gt.to_kind(Kind::Float), None, Reduction::None);
    let positive_loss = &loss * positive.to_kind(Kind::Float);
    let negative_loss_temp = loss * negative.to_kind(Kind::Float);

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
    is_output_polygon: bool,
    dimensions: (u32, u32),
) -> Result<(f64, f64, f64)> {
    let mut raw_metrics = Vec::new();
    let (w, h) = (dimensions.0 as i64, dimensions.1 as i64);
    for i in 0..dataset_paths.test_images.len() {
        debug!("Calculating accuracy for batch {}", i);
        let images = Tensor::load(&dataset_paths.test_images[i])?
            .view((-1, 1, h, w))
            .to_kind(Kind::Float);
        let adjs = Tensor::load(&dataset_paths.test_adj[i])?;
        let polys = image_ops::load_polygons_vec_from_file(&dataset_paths.test_polys[i])?;
        let ignore_flags =
            image_ops::load_ignore_flags_vec_from_file(&dataset_paths.test_ignore_flags[i])?;
        let mut metrics = Vec::with_capacity(images.size()[0] as usize);
        for j in 0..images.size()[0] {
            let inst = Instant::now();
            let mut inference_instant = Instant::now();
            let pred = net.forward_t(&images.get(j).view((-1, 1, h, w)), false);
            let inference_time = inference_instant.elapsed().as_millis();
            inference_instant = Instant::now();
            let (boxes, scores) =
                get_boxes_and_box_scores(&pred, &adjs.get(j).view((-1, 2)), is_output_polygon)?;
            let box_scores_time = inference_instant.elapsed().as_millis();
            inference_instant = Instant::now();
            metrics.push(validate_measure(
                &polys,
                &ignore_flags,
                &boxes,
                &scores,
                j as usize,
            )?);
            let validate_time = inference_instant.elapsed().as_millis();
            debug!(
                "completed validate_measure for image {} in {} ms (inference {} ms, get_box_scores {} ms, validation {} ms",
                j,
                inst.elapsed().as_millis(),
                inference_time, box_scores_time, validate_time
            );
        }
        raw_metrics.push(metrics);
    }
    let metrics = gather_measure(&raw_metrics)?;
    Ok(metrics)
}

fn get_boxes_and_box_scores(
    pred: &Tensor,
    adjust_values: &Tensor,
    is_output_polygon: bool,
) -> Result<(BatchPolygons, Vec<Vec<f64>>)> {
    let thresh = 0.6;
    let mut boxes_batch = Vec::new();
    let mut scores_batch = Vec::new();
    let segmentation = binarize(&pred, thresh)?;

    for batch_index in 0..pred.size()[0] {
        let (boxes, scores) = if is_output_polygon {
            get_polygons_from_bitmap(
                &pred.get(batch_index),
                &segmentation.get(batch_index),
                &adjust_values.get(batch_index),
            )?
        } else {
            get_boxes_from_bitmap(
                &pred.get(batch_index),
                &segmentation.get(batch_index),
                &adjust_values.get(batch_index),
            )?
        };
        boxes_batch.push(boxes);
        scores_batch.push(scores);
    }
    Ok((BatchPolygons(boxes_batch), scores_batch))
}

fn get_polygons_from_bitmap(
    pred: &Tensor,
    bitmap: &Tensor,
    adjust_values: &Tensor,
) -> Result<(MultiplePolygons, Vec<f64>)> {
    let max_candidates = 1000;
    let box_thresh = 0.7;
    let save_pred_to_file = false;

    // convert bitmap to GrayImage
    let image = image_ops::convert_tensor_to_image(&(bitmap.get(0) * 255))?;
    if save_pred_to_file {
        image.save(&format!(
            "text_detection_results/prediction_binarized{}.png",
            COUNTER.lock().unwrap()[0]
        ))?;
        COUNTER.lock().unwrap()[0] += 1;
    }

    let contours = find_contours(&image)
        .into_iter()
        .map(|c| c.points)
        .collect::<Vec<Vec<Point<u32>>>>();
    let num_contours = contours.len().min(max_candidates);
    let mut boxes = vec![image_ops::Polygon { points: vec![] }; num_contours];
    let mut scores = vec![0.; num_contours];

    for i in 0..num_contours {
        let mut epsilon = 0.01 * arc_length(&contours[i], true);
        if epsilon == 0. {
            epsilon = 0.01;
        }
        let mut points = approx_poly_dp(&contours[i], epsilon, true);
        if points.len() > 1 && points[0] == points[points.len() - 1] {
            points.pop();
        }

        if points.len() < 4 {
            continue;
        }
        let score = box_score_fast(&pred.get(0), &points)?;
        if box_thresh > score {
            continue;
        }

        boxes[i].points = points.iter().fold(Vec::new(), |mut acc, p| {
            acc.push(Point::new(
                (p.x as f64 / adjust_values.double_value(&[0])).round() as u32,
                (p.y as f64 / adjust_values.double_value(&[1])).round() as u32,
            ));
            acc
        });
        scores[i] = score;
    }

    Ok((MultiplePolygons(boxes), scores))
}

fn binarize(pred: &Tensor, thresh: f64) -> Result<Tensor> {
    Ok(pred.gt(thresh).to_kind(Kind::Uint8))
}

fn get_boxes_from_bitmap(
    pred: &Tensor,
    bitmap: &Tensor,
    adjust_values: &Tensor,
) -> Result<(MultiplePolygons, Vec<f64>)> {
    let max_candidates = 1000;
    let min_size = 3.;
    let box_thresh = 0.7;

    // convert bitmap to GrayImage
    let image = image_ops::convert_tensor_to_image(&(bitmap.get(0) * 255))?;

    let contours = find_contours(&image)
        .into_iter()
        .map(|c| c.points)
        .collect::<Vec<Vec<Point<u32>>>>();
    let num_contours = contours.len().min(max_candidates);
    let mut boxes = vec![image_ops::Polygon { points: vec![] }; num_contours];
    let mut scores = vec![0.; num_contours];

    for i in 0..num_contours {
        let (points, sside) = get_mini_area_bounding_box(&contours[i]);
        if sside < min_size {
            continue;
        }
        let score = box_score_fast(&pred.get(0), &points)?;
        if box_thresh > score {
            continue;
        }

        boxes[i].points = points.iter().fold(Vec::new(), |mut acc, p| {
            acc.push(Point::new(
                (p.x as f64 / adjust_values.double_value(&[0, 0])).round() as u32,
                (p.y as f64 / adjust_values.double_value(&[0, 1])).round() as u32,
            ));
            acc
        });
        scores[i] = score;
    }

    Ok((MultiplePolygons(boxes), scores))
}

fn get_mini_area_bounding_box(contour: &[Point<u32>]) -> (Vec<Point<u32>>, f64) {
    let mut b_box = min_area_rect(contour);
    b_box.sort_by(|a, b| a.x.cmp(&b.x));
    let i1 = if b_box[1].y > b_box[0].y { 0 } else { 1 };
    let i2 = if b_box[3].y > b_box[2].y { 2 } else { 3 };
    let i3 = if b_box[3].y > b_box[2].y { 3 } else { 2 };
    let i4 = if b_box[1].y > b_box[0].y { 1 } else { 0 };

    let res = vec![b_box[i1], b_box[i2], b_box[i3], b_box[i4]];
    let w = get_distance(&res[0], &res[1]);
    let h = get_distance(&res[0], &res[3]);
    (res, w.min(h))
}

fn box_score_fast(bitmap: &Tensor, points: &[Point<u32>]) -> Result<f64> {
    let size = bitmap.size();
    let w = size[size.len() - 2] as u32;
    let h = size[size.len() - 1] as u32;
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        points
            .iter()
            .fold((std::u32::MAX, 0, std::u32::MAX, 0), |acc, p| {
                (
                    acc.0.min(p.x),
                    acc.1.max(p.x),
                    acc.2.min(p.y),
                    acc.3.max(p.y),
                )
            });
    min_x = clamp(min_x, 0, w - 1);
    max_x = clamp(max_x, 0, w - 1);
    min_y = clamp(min_y, 0, h - 1);
    max_y = clamp(max_y, 0, h - 1);
    let mut mask_image = GrayImage::new((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
    let moved_points: Vec<Point<i32>> = points
        .iter()
        .map(|p| Point::new((p.x - min_x) as i32, (p.y - min_y) as i32))
        .collect();
    drawing::draw_polygon_mut(&mut mask_image, &moved_points, Luma::white());

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
    polygons: &BatchPolygons,
    ignore_tags: &[Vec<bool>],
    pred: &BatchPolygons,
    scores: &[Vec<f64>],
    idx: usize,
    // is_output_polygon: bool,
) -> Result<MetricsItem> {
    let is_output_polygon = true;
    let box_thresh = 0.6;
    let mut pred_polygons = Vec::new();
    for j in 0..pred.0[0].0.len() {
        if is_output_polygon || scores[0][j] >= box_thresh {
            pred_polygons.push(pred.0[0].0[j].clone());
        }
    }

    evaluate_image(
        &polygons.0[idx],
        &ignore_tags[idx],
        &MultiplePolygons(pred_polygons),
    )
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

fn evaluate_image(
    gt_points: &MultiplePolygons,
    ignore_flags: &[bool],
    pred: &MultiplePolygons,
) -> Result<MetricsItem> {
    let area_precision_constraint = 0.5;
    let iou_constraint = 0.5;

    let mut det_matched = 0;

    let mut gt_polys = Vec::new();
    let mut det_polys = Vec::new();

    let mut gt_dont_care_pols_num = Vec::new();
    let mut det_dont_care_pols_num = Vec::new();

    let mut pairs = Vec::new();
    let mut det_matched_nums = Vec::new();

    for (n, curr_gt_points) in gt_points.0.iter().enumerate() {
        let polygon = Polygon::new(
            LineString::from(
                curr_gt_points
                    .points
                    .iter()
                    .map(|p| (p.x as f64, p.y as f64))
                    .collect::<Vec<(f64, f64)>>(),
            ),
            vec![],
        );

        gt_polys.push(polygon);
        if ignore_flags[n] {
            gt_dont_care_pols_num.push(gt_polys.len() - 1);
        }
    }

    for pred_tensor in &pred.0 {
        let polygon = Polygon::new(
            LineString::from(
                pred_tensor
                    .points
                    .iter()
                    .map(|p| (p.x as f64, p.y as f64))
                    .collect::<Vec<(f64, f64)>>(),
            ),
            vec![],
        );
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
        let res = get_mini_area_bounding_box(&[
            Point::new(141, 24),
            Point::new(61, 16),
            Point::new(57, 53),
            Point::new(137, 61),
        ]);
        assert_eq!(
            res.0,
            vec![
                Point::new(60, 15),
                Point::new(142, 23),
                Point::new(138, 62),
                Point::new(57, 54)
            ]
        );
        assert!(res.1 - 39.11521443121589 < ERROR_THRESHOLD);
    }

    #[test]
    fn box_score_with_whole_matrix_test() -> Result<()> {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![
            Point::new(0, 0),
            Point::new(4, 0),
            Point::new(4, 4),
            Point::new(0, 4),
        ];
        assert_eq!(box_score_fast(&pred, &points)?, 10. / 25.);
        Ok(())
    }

    #[test]
    fn box_score_partial_matrix_test() -> Result<()> {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![
            Point::new(1, 0),
            Point::new(4, 0),
            Point::new(4, 3),
            Point::new(1, 3),
        ];
        assert_eq!(box_score_fast(&pred, &points)?, 8. / 16.);
        Ok(())
    }

    #[test]
    fn box_score_partial_matrix_test_2() -> Result<()> {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![
            Point::new(2, 0),
            Point::new(4, 1),
            Point::new(2, 4),
            Point::new(1, 3),
        ];
        assert_eq!(box_score_fast(&pred, &points)?, 9. / 12.);
        Ok(())
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
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(235, 17),
                Point::new(302, 21),
                Point::new(299, 59),
                Point::new(233, 54),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(7, 75),
                Point::new(148, 95),
                Point::new(145, 111),
                Point::new(4, 91),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(250, 79),
                Point::new(303, 85),
                Point::new(299, 126),
                Point::new(245, 120),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(25, 181),
                Point::new(125, 181),
                Point::new(125, 268),
                Point::new(25, 268),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(190, 250),
                Point::new(255, 185),
                Point::new(285, 215),
                Point::new(220, 280),
            ],
            Luma::white(),
        );
        let pred_tensor = (convert_image_to_tensor(&prediction_image)? / 255.).view((1, h, w));

        // 2x adjustment
        let adjust_values = Tensor::of_slice(&[2., 2.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            // expected boxes are reajdusted to the original image size (divided by adjust values)
            image_ops::Polygon {
                points: vec![
                    Point::new(30, 8),
                    Point::new(71, 12),
                    Point::new(69, 31),
                    Point::new(29, 27),
                ],
            }, // 1
            image_ops::Polygon {
                points: vec![
                    Point::new(118, 9),
                    Point::new(151, 11),
                    Point::new(150, 30),
                    Point::new(117, 27),
                ],
            }, // 2
            image_ops::Polygon {
                points: vec![
                    Point::new(4, 38),
                    Point::new(74, 48),
                    Point::new(73, 56),
                    Point::new(2, 46),
                ],
            }, // 3
            image_ops::Polygon {
                points: vec![
                    Point::new(125, 40),
                    Point::new(152, 43),
                    Point::new(150, 63),
                    Point::new(123, 60),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(13, 91),
                    Point::new(63, 91),
                    Point::new(63, 134),
                    Point::new(13, 134),
                ],
            }, // 5
            image_ops::Polygon {
                points: vec![
                    Point::new(95, 125),
                    Point::new(128, 93),
                    Point::new(143, 108),
                    Point::new(110, 140),
                ],
            }, // 6
        ]);
        let expected_scores = vec![1.; 6];
        assert_eq!(
            get_boxes_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_boxes_from_bitmap_test_partial_match_without_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // no adjustment
        let adjust_values = Tensor::of_slice(&[1., 1.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            // expected boxes with score over thresh (0.7) should appear
            image_ops::Polygon { points: vec![] }, // 1
            image_ops::Polygon { points: vec![] }, // 2
            image_ops::Polygon { points: vec![] }, // 3
            image_ops::Polygon {
                points: vec![
                    Point::new(250, 79),
                    Point::new(303, 85),
                    Point::new(299, 126),
                    Point::new(245, 120),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(25, 181),
                    Point::new(125, 181),
                    Point::new(125, 268),
                    Point::new(25, 268),
                ],
            }, // 5
            image_ops::Polygon { points: vec![] }, // 6
        ]);
        let expected_scores = vec![0., 0., 0., 0.703405017921147, 0.7521377137713772, 0.];
        assert_eq!(
            get_boxes_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_polygons_from_bitmap_test_without_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // no adjustment
        let adjust_values = Tensor::of_slice(&[1., 1.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(100, 20),
                    Point::new(90, 35),
                    Point::new(60, 25),
                    Point::new(90, 40),
                    Point::new(80, 55),
                    Point::new(101, 50),
                    Point::new(130, 60),
                    Point::new(115, 45),
                    Point::new(140, 30),
                    Point::new(120, 35),
                ],
            }, // 1
            image_ops::Polygon {
                points: vec![
                    Point::new(275, 20),
                    Point::new(265, 35),
                    Point::new(235, 25),
                    Point::new(265, 40),
                    Point::new(255, 55),
                    Point::new(276, 50),
                    Point::new(299, 58),
                    Point::new(299, 54),
                    Point::new(290, 45),
                    Point::new(299, 40),
                    Point::new(299, 34),
                    Point::new(295, 35),
                ],
            }, // 2
            image_ops::Polygon { points: vec![] }, // 3, it is ignored since its a triangle
            image_ops::Polygon {
                points: vec![
                    Point::new(250, 80),
                    Point::new(250, 120),
                    Point::new(296, 125),
                    Point::new(299, 125),
                    Point::new(299, 104),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(50, 181),
                    Point::new(25, 225),
                    Point::new(49, 268),
                    Point::new(100, 268),
                    Point::new(125, 225),
                    Point::new(100, 181),
                ],
            }, // 5
            image_ops::Polygon {
                points: vec![
                    Point::new(269, 200),
                    Point::new(240, 210),
                    Point::new(190, 250),
                    Point::new(220, 280),
                ],
            }, // 6
        ]);
        let expected_scores = vec![
            1.0,
            0.9980139026812314,
            0.0,
            0.9924146649810367,
            1.0,
            0.9978822532825075,
        ];
        assert_eq!(
            get_polygons_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_polygons_from_bitmap_test_with_2x_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // resized by w x 2, h x 2
        let adjust_values = Tensor::of_slice(&[2., 2.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(50, 10),
                    Point::new(45, 18),
                    Point::new(30, 13),
                    Point::new(45, 20),
                    Point::new(40, 28),
                    Point::new(51, 25),
                    Point::new(65, 30),
                    Point::new(58, 23),
                    Point::new(70, 15),
                    Point::new(60, 18),
                ],
            }, // 1
            image_ops::Polygon {
                points: vec![
                    Point::new(138, 10),
                    Point::new(133, 18),
                    Point::new(118, 13),
                    Point::new(133, 20),
                    Point::new(128, 28),
                    Point::new(138, 25),
                    Point::new(150, 29),
                    Point::new(150, 27),
                    Point::new(145, 23),
                    Point::new(150, 20),
                    Point::new(150, 17),
                    Point::new(148, 18),
                ],
            }, // 2
            image_ops::Polygon { points: vec![] }, // 3, it is ignored since its a triangle
            image_ops::Polygon {
                points: vec![
                    Point::new(125, 40),
                    Point::new(125, 60),
                    Point::new(148, 63),
                    Point::new(150, 63),
                    Point::new(150, 52),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(25, 91),
                    Point::new(13, 113),
                    Point::new(25, 134),
                    Point::new(50, 134),
                    Point::new(63, 113),
                    Point::new(50, 91),
                ],
            }, // 5
            image_ops::Polygon {
                points: vec![
                    Point::new(135, 100),
                    Point::new(120, 105),
                    Point::new(95, 125),
                    Point::new(110, 140),
                ],
            }, // 6
        ]);
        let expected_scores = vec![
            1.0,
            0.9980139026812314,
            0.0,
            0.9924146649810367,
            1.0,
            0.9978822532825075,
        ];
        assert_eq!(
            get_polygons_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn evaluate_image_test_one_matching_polygon() -> Result<()> {
        let gt_polygons_tensor = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(0, 0),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            }, // poly 1
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ],
            }, // poly 2
        ]);
        let ignore_polygons = [false, false];
        let pred = MultiplePolygons(vec![image_ops::Polygon {
            points: vec![
                Point::new(1, 1),
                Point::new(10, 0),
                Point::new(10, 10),
                Point::new(0, 10),
            ],
        }]);
        let metrics = evaluate_image(&gt_polygons_tensor, &ignore_polygons, &pred)?;

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
        let gt_polygons_tensor = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(0, 0),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            }, // poly 1
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ], // poly 2
            },
        ]);
        let ignore_polygons = [true, true];
        let pred = MultiplePolygons(vec![image_ops::Polygon {
            points: vec![
                Point::new(1, 1),
                Point::new(10, 0),
                Point::new(10, 10),
                Point::new(0, 10),
            ],
        }]);
        let metrics = evaluate_image(&gt_polygons_tensor, &ignore_polygons, &pred)?;

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
        let gt_polygons_tensor = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(0, 0),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            }, // poly 1
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ],
            }, // poly 2
        ]);
        let ignore_polygons = [false, false];
        let pred = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(1, 1),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            },
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ],
            },
        ]);
        let metrics = evaluate_image(&gt_polygons_tensor, &ignore_polygons, &pred)?;

        assert_eq!(metrics.gt_care, 2);
        assert_eq!(metrics.det_care, 2);
        assert_eq!(metrics.det_matched, 2);
        assert!(metrics.precision - 1. < ERROR_THRESHOLD);
        assert!(metrics.recall - 1. < ERROR_THRESHOLD);
        assert!(metrics.hmean - 1. < ERROR_THRESHOLD);

        Ok(())
    }

    // #[test]
    // fn validate_measure_test() -> Result<()> {
    //     let gt_polygons_tensor = BatchPolygons(vec![
    //         // image 1
    //         MultiplePolygons(vec![
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(0, 0),
    //                     Point::new(10, 0),
    //                     Point::new(10, 10),
    //                     Point::new(0, 10),
    //                 ],
    //             }, // poly 1
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(20, 20),
    //                     Point::new(30, 20),
    //                     Point::new(30, 30),
    //                     Point::new(20, 30),
    //                 ],
    //             }, // poly 2
    //         ]),
    //         // image 2
    //         MultiplePolygons(vec![
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(0, 0),
    //                     Point::new(10, 0),
    //                     Point::new(10, 10),
    //                     Point::new(0, 10),
    //                 ],
    //             }, // poly 1
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(20, 20),
    //                     Point::new(30, 20),
    //                     Point::new(30, 30),
    //                     Point::new(20, 30),
    //                 ],
    //             }, // poly 2
    //         ]),
    //     ]);
    //     let ignore_polygons = [
    //         vec![false, false], // image 1
    //         vec![false, false], // image 2
    //     ];
    //     let pred = BatchPolygons(vec![
    //         MultiplePolygons(vec![image_ops::Polygon {
    //             points: vec![
    //                 Point::new(1, 1),
    //                 Point::new(10, 0),
    //                 Point::new(10, 10),
    //                 Point::new(0, 10),
    //             ],
    //         }]),
    //         MultiplePolygons(vec![image_ops::Polygon {
    //             points: vec![
    //                 Point::new(45, 61),
    //                 Point::new(47, 41),
    //                 Point::new(60, 60),
    //                 Point::new(39, 48),
    //             ],
    //         }]),
    //     ]);
    //     let scores = [vec![0.9], vec![0.9]];
    //     let metrics = validate_measure(&gt_polygons_tensor, &ignore_polygons, &pred, &scores)?;
    //     assert_eq!(metrics.len(), 2);

    //     // image 1 metrics (having 1 matched polygon)
    //     assert_eq!(metrics[0].gt_care, 2);
    //     assert_eq!(metrics[0].det_care, 1);
    //     assert_eq!(metrics[0].det_matched, 1);
    //     assert!(metrics[0].precision - 1. < ERROR_THRESHOLD);
    //     assert!(metrics[0].recall - 0.5 < ERROR_THRESHOLD);
    //     assert!(metrics[0].hmean - 0.6666666666666666 < ERROR_THRESHOLD);

    //     // image 2 metrics (having 0 matched polygons)
    //     assert_eq!(metrics[1].gt_care, 2);
    //     assert_eq!(metrics[1].det_care, 1);
    //     assert_eq!(metrics[1].det_matched, 0);
    //     assert!(metrics[1].precision < ERROR_THRESHOLD);
    //     assert!(metrics[1].recall < ERROR_THRESHOLD);
    //     assert!(metrics[1].hmean < ERROR_THRESHOLD);
    //     Ok(())
    // }

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
