#[macro_use]
extern crate lazy_static;
extern crate clap;
extern crate log;
extern crate log4rs;
extern crate tch;

mod char_recognition;
mod dataset;
mod image_ops;
#[macro_use]
mod macros;
mod text_detection;
mod utils;

use anyhow::Result;
use char_recognition::options::CharRecOptions;
use clap::{App, AppSettings, SubCommand};
use tch::Device;
use text_detection::{options::TextDetOptions, DEFAULT_HEIGHT, DEFAULT_WIDTH};

const CHAR_REC_SC: &str = "char-recognition";
const TEXT_DETECTION_SC: &str = "text-detection";

lazy_static! {
    pub static ref DEVICE: Device = Device::cuda_if_available();
}

fn main() -> Result<()> {
    log4rs::init_file("log4rs.yml", Default::default())?;

    let matches = App::new("CLI for ocr-rs")
        .version("0.0.1")
        .author("Zoran Lazarevic <lazarevic.zoki91@gmail.com>")
        .about("A CLI tool for training/evaluating models for character recognition/text detection/OCR")
        .subcommand(
            SubCommand::with_name(CHAR_REC_SC)
                .alias("cr")
                .about("Controls for the character recognition model")
                .setting(AppSettings::SubcommandsNegateReqs)
                .args_from_usage(
                    "<INPUT>                'Path to the file to run the detection on'
                    --model-file [FILE]     'Path to the model file'"
                )
                .subcommand(
                    SubCommand::with_name("train").about("Trains the model")
                        .args_from_usage(
                            "--model-file [FILE]        'Path to the model file'
                            --tensor-files-dir [DIR]    'Path to the tensor files directory'
                            --resume                    'Continue the training process on an existing model'
                            --epoch [EPOCH]             'Specify the number of training iterations'
                            --learning-rate [LR]        'Specify the learning rate'"
                        )
                )
                .subcommand(
                    SubCommand::with_name("prepare-tensors").about("Converts images/labels into tensors and stores them on the file system")
                        .args_from_usage(
                            "--target-dir [DIR]         'Path to the directory where the files will be stored (default: .)'
                            --data-dir [DATA_DIR]       'Path to the directory containing the images need to be loaded (default: ./images)'"
                        )
                )
        )
        .subcommand(
            SubCommand::with_name(TEXT_DETECTION_SC)
                .alias("td")
                .about("Controls for the text detection model training/inference")
                .setting(AppSettings::SubcommandsNegateReqs)
                .args_from_usage(
                    "<INPUT>                'Path to the file to run the detection on'
                    --model-file [FILE]     'Path to the model file'
                    -d, --dimensions [DIM]  'The target dimensions of preprocessed image (e.g. 800x800)'"
                )
                .subcommand(
                    SubCommand::with_name("train").about("Trains the model")
                        .args_from_usage(
                            "--model-file [FILE]        'Path to the model file'
                            --tensor-files-dir [DIR]    'Path to the tensor files directory'
                            --resume                    'Continue the training process on an existing model'
                            --epoch [EPOCH]             'Specify the number of training iterations'
                            --learning-rate [LR]        'Specify the learning rate'
                            --momentum [M]              'Specify the SGD momentum'
                            --dampening [D]             'Specify the SGD dampening'
                            --weight-decay [WD]         'Specify the SGD weight decay'
                            --no-nesterov               'Do not use Nesterov momentum'
                            -d, --dimensions [DIM]      'The target dimensions of preprocessed image (e.g. 800x800)'
                            --test-interval [INT]       'Specify the the number of iterations after which the accuracy test will run'"
                        )
                )
                .subcommand(
                    SubCommand::with_name("prepare-tensors").about("Converts images/gts into tensors and stores them on the file system")
                        .args_from_usage(
                            "--target-dir [DIR]         'Path to the directory where the files will be stored (default: .)'
                            --data-dir [DATA_DIR]       'Path to the directory containing the images need to be loaded (default: ./text-detection-images)'
                            -d, --dimensions [DIM]      'The target dimensions of preprocessed image (e.g. 800x800)'"
                        )
                )
        )
        .get_matches();

    if let Some(scmd_matches) = matches.subcommand_matches(CHAR_REC_SC) {
        // handling character recognition match
        if let Some(train_mathes) = scmd_matches.subcommand_matches("train") {
            let options = CharRecOptions::new(train_mathes)?;

            char_recognition::train_model(&options)?;
        } else if let Some(prep_matches) = scmd_matches.subcommand_matches("prepare-tensors") {
            let data_dir = prep_matches
                .value_of("data-dir")
                .unwrap_or(image_ops::CHAR_REC_IMAGES_PATH);
            let target_dir = prep_matches.value_of("target-dir").unwrap_or(".");

            image_ops::generate_char_rec_tensor_files(data_dir, target_dir)?;
        } else {
            let image_path = scmd_matches.value_of("INPUT").unwrap();
            let model_file_path = scmd_matches
                .value_of("model-file")
                .unwrap_or(char_recognition::MODEL_FILENAME);

            char_recognition::run_prediction(image_path, model_file_path)?;
        }
    } else if let Some(scmd_matches) = matches.subcommand_matches(TEXT_DETECTION_SC) {
        // handling text detection match
        if let Some(train_mathes) = scmd_matches.subcommand_matches("train") {
            let options = TextDetOptions::new(train_mathes)?;

            text_detection::train_model(&options)?;
        } else if let Some(prep_matches) = scmd_matches.subcommand_matches("prepare-tensors") {
            let data_dir = prep_matches
                .value_of("data-dir")
                .unwrap_or(image_ops::TEXT_DET_IMAGES_PATH);
            let target_dir = prep_matches.value_of("target-dir").unwrap_or(".");

            let mut dimensions = (DEFAULT_WIDTH, DEFAULT_HEIGHT);
            if let Some(dims_str) = scmd_matches.value_of("dimensions") {
                dimensions = utils::parse_dimensions(dims_str)?;
            }

            image_ops::generate_text_det_tensor_chunks(data_dir, target_dir, true, dimensions)?;
            image_ops::generate_text_det_tensor_chunks(data_dir, target_dir, false, dimensions)?;
        } else {
            let image_path = scmd_matches.value_of("INPUT").unwrap();
            let model_file_path = scmd_matches
                .value_of("model-file")
                .unwrap_or(text_detection::MODEL_FILENAME);

            let mut dimensions = (DEFAULT_WIDTH, DEFAULT_HEIGHT);
            if let Some(dims_str) = scmd_matches.value_of("dimensions") {
                dimensions = utils::parse_dimensions(dims_str)?;
            }

            text_detection::run_text_detection(image_path, model_file_path, dimensions)?;
        }
    }

    Ok(())
}
