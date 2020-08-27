#[macro_use]
extern crate lazy_static;
extern crate log;
extern crate log4rs;
extern crate tch;

mod char_rec_conv_nn;
mod char_rec_nn;
mod image_ops;
mod utils;

use anyhow::Result;

fn main() -> Result<()> {
    log4rs::init_file("log4rs.yml", Default::default())?;

    let image_tensor =
        image_ops::load_image_as_tensor("images/test/akronim-dist-1-regular-upper-M-img.png")?;
    char_rec_conv_nn::run_prediction(&image_tensor)?;
    // char_rec_nn::run_prediction(&image_tensor)?;

    Ok(())
}
