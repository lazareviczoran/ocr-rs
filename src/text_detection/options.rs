use super::{DEFAULT_HEIGHT, DEFAULT_TENSORS_DIR, DEFAULT_WIDTH, MODEL_FILENAME};
use crate::utils::{parse_dimensions, parse_number};
use anyhow::Result;

#[derive(Debug)]
pub struct TextDetOptions<'a> {
    pub epoch: usize,
    pub model_file_path: &'a str,
    pub tensor_files_dir: &'a str,
    pub learning_rate: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub image_dimensions: (u32, u32),
    pub test_interval: usize,
    pub resume: bool,
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
