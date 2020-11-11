pub mod model;
pub mod options;

use super::image_ops;
use super::measure_time;
use super::utils::{save_vs, topk};
use super::DEVICE;
use anyhow::{anyhow, Result};
use log::{debug, info};
use model::Net;
use options::CharRecOptions;
use std::path::Path;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Kind};

pub const MODEL_FILENAME: &str = "char_rec_conv_net.model";

pub fn train_model(opts: &CharRecOptions) -> Result<Net> {
    let m = image_ops::load_values(opts.tensor_files_dir)?;
    let vs = nn::VarStore::new(*DEVICE);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, opts.learning_rate)?;
    for epoch in 1..=opts.epoch {
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        info!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    save_vs(&vs, opts.model_file_path)?;

    Ok(net)
}

pub fn run_prediction<T: AsRef<Path>>(image_path: T, model_file_path: T) -> Result<()> {
    let path = model_file_path.as_ref();
    if !path.exists() {
        return Err(anyhow!("Model file {} doesn't exist", path.display()));
    }
    let mut weights = nn::VarStore::new(*DEVICE);
    let net = Net::new(&weights.root());
    weights.load(model_file_path)?;

    let image_tensor = image_ops::load_image_as_tensor(image_path)?;
    // recognize character
    let (predicted_value, probability) = measure_time!(
        "character classification",
        || {
            let res = net
                .forward_t(&image_tensor, false)
                .softmax(-1, Kind::Double);
            topk(&res, 1)[0]
        },
        LogType::Debug
    );

    debug!(
        "The image is classified as {} with {:3.2}% of certainty",
        predicted_value,
        probability * 100.
    );

    Ok(())
}
