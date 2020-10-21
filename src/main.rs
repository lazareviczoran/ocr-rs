#[macro_use]
extern crate lazy_static;
extern crate clap;
extern crate geo;
extern crate geo_booleanop;
extern crate log;
extern crate log4rs;
extern crate tch;

mod char_rec_conv_nn;
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
                // .arg(
                //     Arg::with_name("debug")
                //         .short("d")
                //         .help("print debug information verbosely"),
                // ),
        )
        .subcommand(
            SubCommand::with_name("text-detection")
                .about("Controls for the text detection model")
                // .arg(
                //     Arg::with_name("debug")
                //         .short("d")
                //         .help("print debug information verbosely"),
                // ),
        )
        .subcommand(
            SubCommand::with_name("image-ops")
                .about("Offers some useful image operations")
                // .arg(
                //     Arg::with_name("debug")
                //         .short("d")
                //         .help("print debug information verbosely"),
                // ),
        )
        .get_matches();

    if let Some(_matches) = matches.subcommand_matches(CHAR_REC_SC) {
        let image_tensor =
            image_ops::load_image_as_tensor("images/test/akronim-dist-1-regular-upper-M-img.png")?;
        char_rec_conv_nn::run_prediction(&image_tensor)?;
    } else if let Some(_matches) = matches.subcommand_matches(TEXT_DETECTION_SC) {
        // text_detection_model::run_text_detection("Screenshot_20200726-134601_Firefox.jpg")?;
        text_detection_model::create_and_train_model()?;
    } else if let Some(_matches) = matches.subcommand_matches(IMAGE_OPS_SC) {
        // image_ops::preprocess_image("Screen Shot 2020-08-27 at 5.37.59 PM.png")?;
        // image_ops::load_text_detection_images()?;
        image_ops::generate_text_det_tensor_chunks(image_ops::TEXT_DET_IMAGES_PATH, true, None)?;
        image_ops::generate_text_det_tensor_chunks(image_ops::TEXT_DET_IMAGES_PATH, false, None)?;
    }

    Ok(())
}
