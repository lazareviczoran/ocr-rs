pub mod metrics;
pub mod model;
pub mod options;

use super::dataset::TextDetectionDataset;
use super::image_ops;
use super::{measure_time, DEVICE};
use crate::utils::save_vs;
use anyhow::{anyhow, Result};
use itertools::izip;
use log::{debug, info};
use metrics::{gather_measure, get_boxes_and_box_scores, validate_measure, PolygonScores};
use model::resnet18;
use options::TextDetOptions;
use std::path::Path;
use tch::{nn, nn::FuncT, nn::ModuleT, nn::OptimizerConfig, Kind, Reduction, Tensor};

pub const MODEL_FILENAME: &str = "text_detection.model";
const DEFAULT_TENSORS_DIR: &str = "./text_det_tensor_files";
pub const DEFAULT_WIDTH: u32 = 800;
pub const DEFAULT_HEIGHT: u32 = 800;

pub fn run_text_detection<T: AsRef<std::path::Path>>(
    image_path: T,
    model_file_path: T,
    dimensions: (u32, u32),
) -> Result<()> {
    let (preprocessed_image, adj_x, adj_y) = measure_time!(
        "image preprocessing",
        || -> Result<(image::GrayImage, f64, f64)> {
            let val = image_ops::preprocess_image(image_path, dimensions)?;
            Ok(val)
        }
    )?;
    let mut vs = nn::VarStore::new(*DEVICE);
    let path = model_file_path.as_ref();
    if !path.exists() {
        return Err(anyhow!("Model file {} doesn't exist", path.display()));
    }
    let net = resnet18(&vs.root());
    measure_time!("loading model", || -> Result<()> {
        vs.load(model_file_path)?;
        Ok(())
    })?;

    let image_tensor = measure_time!("image -> tensor", || -> Result<Tensor> {
        let tensor = image_ops::convert_image_to_tensor(&preprocessed_image)?.to_kind(Kind::Float);
        Ok(tensor)
    })?;

    let (w, h) = dimensions;
    let pred = measure_time!("inference", || {
        net.forward_t(&image_tensor.view((1, 1, h as i64, w as i64)), false)
    });

    let pred_image = measure_time!("tensor -> image", || -> Result<image::GrayImage> {
        let image = image_ops::convert_tensor_to_image(&(pred.get(0).get(0) * 255.))?;
        Ok(image)
    })?;
    pred_image.save("pred.png")?;
    let PolygonScores {
        polygons: _boxes,
        scores: _scores,
    } = measure_time!("found contours", || -> Result<PolygonScores> {
        let res = get_boxes_and_box_scores(&pred, &Tensor::of_slice(&[adj_x, adj_y]).view((1, 2)))?;
        Ok(res)
    })?;
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
        measure_time!("training single batch", || -> Result<()> {
            for (i, images_batch) in dataset_paths.train_images.iter().enumerate() {
                debug!("data batch: {}", i);
                let image = Tensor::load(&images_batch)?
                    .view((-1, 1, h, w))
                    .to_kind(Kind::Float);
                let num_of_chunks = (image.size()[0] as f32 / opts.chunk_size as f32).ceil() as i64;
                let gt = Tensor::load(&dataset_paths.train_gt[i])?;
                let mask = Tensor::load(&dataset_paths.train_mask[i])?;
                for (images_chunk, gt_chunk, mask_chunk) in izip!(
                    image.chunk(num_of_chunks, 0).iter(),
                    gt.chunk(num_of_chunks, 0).iter(),
                    mask.chunk(num_of_chunks, 0).iter()
                ) {
                    let pred = measure_time!("single training pred", || {
                        net.forward_t(&images_chunk, true)
                    });
                    let loss = measure_time!("single training loss", || {
                        calculate_balance_loss(&pred, gt_chunk, mask_chunk)
                    });
                    measure_time!("single training backward step", || opt.backward_step(&loss));
                }
                save_vs(&vs, opts.model_file_path)?;
            }
            Ok(())
        })?;
        if epoch % opts.test_interval == 0 || epoch == epoch_limit {
            let (precision, _recall, _hmean) =
                measure_time!("test process", || -> Result<(f64, f64, f64)> {
                    let res = get_model_accuracy(&dataset_paths, &net, opts)?;
                    Ok(res)
                })?;
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
fn calculate_balance_loss(pred: &Tensor, gt: &Tensor, mask: &Tensor) -> Tensor {
    let negative_ratio = 3.0;
    let eps = 1e-6;

    let positive = gt * mask;
    let negative: Tensor = (1 - gt) * mask;
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
    opts: &TextDetOptions,
) -> Result<(f64, f64, f64)> {
    let mut raw_metrics = Vec::new();
    let (w, h) = (
        opts.image_dimensions.0 as i64,
        opts.image_dimensions.1 as i64,
    );
    for i in 0..dataset_paths.test_images.len() {
        debug!("Calculating accuracy for batch {}", i);
        let images = Tensor::load(&dataset_paths.test_images[i])?
            .view((-1, 1, h, w))
            .to_kind(Kind::Float);
        let num_of_chunks = (images.size()[0] as f32 / opts.chunk_size as f32).ceil() as i64;
        let adjs = Tensor::load(&dataset_paths.test_adj[i])?;
        let polys = image_ops::load_polygons_vec_from_file(&dataset_paths.test_polys[i])?;
        let ignore_flags =
            image_ops::load_ignore_flags_vec_from_file(&dataset_paths.test_ignore_flags[i])?;
        for (images_chunk, adjs_chunk, polys_chunk, flags_chunk) in izip!(
            images.chunk(num_of_chunks, 0).iter(),
            adjs.chunk(num_of_chunks, 0).iter(),
            polys.chunks(opts.chunk_size),
            ignore_flags.chunks(opts.chunk_size)
        ) {
            debug!("starting validate_measure for chunk");
            let pred = measure_time!("accuracy calc -> inference", || net
                .forward_t(&images_chunk, false));
            let PolygonScores {
                polygons: boxes,
                scores,
            } = measure_time!("box_scores_time", || -> Result<PolygonScores> {
                let res = get_boxes_and_box_scores(&pred, &adjs_chunk)?;
                Ok(res)
            })?;
            raw_metrics.push(measure_time!(
                "validation",
                || -> Result<Vec<metrics::MetricsItem>> {
                    let res = validate_measure(&polys_chunk, &flags_chunk, &boxes, &scores)?;
                    Ok(res)
                }
            )?);
            debug!("completed validate_measure for image chunk");
        }
    }
    let metrics = gather_measure(&raw_metrics)?;
    Ok(metrics)
}
