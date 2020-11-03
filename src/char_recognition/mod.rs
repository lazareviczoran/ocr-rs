use super::image_ops;
use super::utils::{parse_number, save_vs, topk, VALUES_COUNT_I64};
use super::DEVICE;
use anyhow::{anyhow, Result};
use log::{debug, info};
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Kind, Tensor};

pub const MODEL_FILENAME: &str = "char_rec_conv_net.model";

#[derive(Debug)]
pub struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 512, Default::default());
        let fc2 = nn::linear(vs, 512, VALUES_COUNT_I64, Default::default());
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply_t(&self.conv1, train)
            .max_pool2d_default(2)
            .apply_t(&self.conv2, train)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply_t(&self.fc1, train)
            .relu()
            .dropout_(0.5, train)
            .apply_t(&self.fc2, train)
    }
}

#[derive(Debug)]
pub struct CharRecOptions<'a> {
    epoch: usize,
    model_file_path: &'a str,
    tensor_files_dir: &'a str,
    learning_rate: f64,
    resume: bool,
}

impl Default for CharRecOptions<'_> {
    fn default() -> Self {
        Self {
            epoch: 100,
            model_file_path: MODEL_FILENAME,
            tensor_files_dir: ".",
            learning_rate: 1e-4,
            resume: false,
        }
    }
}
impl<'a> CharRecOptions<'a> {
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
        if args.is_present("resume") {
            opts.resume = true;
        }

        Ok(opts)
    }
}

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

pub fn run_prediction(image_path: &str, model_file_path: &str) -> Result<()> {
    if !Path::new(model_file_path).exists() {
        return Err(anyhow!("Model file {} doesn't exist", model_file_path));
    }
    let mut weights = nn::VarStore::new(*DEVICE);
    let net = Net::new(&weights.root());
    weights.load(model_file_path)?;

    let image_tensor = image_ops::load_image_as_tensor(image_path)?;
    // recognize character
    let instant = Instant::now();
    let res = net
        .forward_t(&image_tensor, false)
        .softmax(-1, Kind::Double);
    let (predicted_value, probability) = topk(&res, 1)[0];

    debug!(
        "finished classification in {:?} ns, with {} as result with {:3.2}% of certainty",
        instant.elapsed().as_nanos(),
        predicted_value,
        probability * 100.
    );

    Ok(())
}
