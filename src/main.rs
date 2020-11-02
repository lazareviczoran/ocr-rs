#[macro_use]
extern crate lazy_static;
extern crate clap;
extern crate geo;
extern crate geo_booleanop;
extern crate log;
extern crate log4rs;
extern crate tch;

mod char_recognition;
mod dataset;
mod image_ops;
mod text_detection_model;
mod utils;

use anyhow::Result;
use clap::{App, SubCommand};
use tch::Device;

const CHAR_REC_SC: &str = "char-rec";
const TEXT_DETECTION_SC: &str = "text-detection";
const IMAGE_OPS_SC: &str = "image-ops";

lazy_static! {
    pub static ref DEVICE: Device = Device::cuda_if_available();
}

fn main() -> Result<()> {
    log4rs::init_file("log4rs.yml", Default::default())?;

    let matches = App::new("CLI for ocr-rs")
        .version("0.0.1")
        .author("Zoran Lazarevic <lazarevic.zoki91@gmail.com>")
        .about("A CLI tool for training/evaluating models for character recognition/text detection/both")
        .subcommand(
            SubCommand::with_name("char-rec")
                .about("Controls for the character recognition model")
        )
        .subcommand(
            SubCommand::with_name("text-detection")
                .about("Controls for the text detection model")
                .subcommand(
                    SubCommand::with_name("run").about("Runs the text detection on the provided image")
                        .args_from_usage("
                        <INPUT>                 'Path to the file to run the detection on'
                        --model-file [FILE]     'Path to the model file'
                        –-dimensions [DIM]      'The target dimensions of preprocessed image (e.g. 800x800)'
                    ")
                )
                .subcommand(
                    SubCommand::with_name("train").about("Trains the model")
                        .args_from_usage("
                            --model-file [FILE]         'Path to the model file'
                            --tensor-files-dir [DIR]    'Path to the tensor files directory'
                            --resume                    'Continue the training process on an existing model'
                            --epoch [EPOCH]             'Specify the number of training iterations'
                            --learning-rate [LR]        'Specify the learning rate'
                            --momentum [M]              'Specify the SGD momentum'
                            --dampening [D]             'Specify the SGD dampening'
                            --weight-decay [WD]         'Specify the SGD weight decay'
                            --no-nesterov               'Do not use Nesterov momentum'
                            --dimensions [DIM]          'The target dimensions of preprocessed image (e.g. 800x800)'
                            --test-interval [INT]       'Specify the the number of iterations after which the accuracy test will run'
                        ")
                ),
        )
        .subcommand(
            SubCommand::with_name("image-ops")
                .about("Offers some useful image operations")
        )
        .get_matches();

    if let Some(scmd_matches) = matches.subcommand_matches(CHAR_REC_SC) {
        // handling character recognition match
        let image_tensor =
            image_ops::load_image_as_tensor("images/test/akronim-dist-1-regular-upper-M-img.png")?;
        char_recognition::run_prediction(&image_tensor)?;
    } else if let Some(scmd_matches) = matches.subcommand_matches(TEXT_DETECTION_SC) {
        // handling text detection match
        if let Some(train_mathes) = scmd_matches.subcommand_matches("train") {
            let options = text_detection_model::TextDetOptions::new(train_mathes)?;
            text_detection_model::train_model(&options)?;
        } else if let Some(run_matches) = scmd_matches.subcommand_matches("run") {
            let image_path = run_matches.value_of("INPUT").unwrap();
            let model_file_path = run_matches
                .value_of("model-file")
                .unwrap_or(text_detection_model::MODEL_FILENAME);
            let mut dimensions = (
                text_detection_model::DEFAULT_WIDTH,
                text_detection_model::DEFAULT_HEIGHT,
            );
            if let Some(dims_str) = run_matches.value_of("dimensions") {
                dimensions = text_detection_model::parse_dimensions(dims_str)?;
            }
            text_detection_model::run_text_detection(image_path, model_file_path, dimensions)?;
        }
    } else if let Some(_matches) = matches.subcommand_matches(IMAGE_OPS_SC) {
        // image_ops::load_text_detection_images()?;
        image_ops::generate_text_det_tensor_chunks(
            image_ops::TEXT_DET_IMAGES_PATH,
            true,
            (800, 800),
        )?;
        image_ops::generate_text_det_tensor_chunks(
            image_ops::TEXT_DET_IMAGES_PATH,
            false,
            (800, 800),
        )?;
    }

    Ok(())
}
