#[macro_use]
extern crate lazy_static;
extern crate log;
extern crate log4rs;
extern crate tch;

mod char_rec_conv_nn;
mod char_rec_nn;
mod image_ops;

use anyhow::Result;
use log::info;

fn main() -> Result<()> {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();

    info!("Started new learning process");
    char_rec_conv_nn::run()?;
    // char_rec_nn::run()?;
    Ok(())
}
