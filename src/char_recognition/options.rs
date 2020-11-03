use super::MODEL_FILENAME;
use crate::utils::parse_number;
use anyhow::Result;

#[derive(Debug)]
pub struct CharRecOptions<'a> {
    pub epoch: usize,
    pub model_file_path: &'a str,
    pub tensor_files_dir: &'a str,
    pub learning_rate: f64,
    pub resume: bool,
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
