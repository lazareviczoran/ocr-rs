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

    // image_ops::preprocess_image("Screen Shot 2020-08-27 at 5.37.59 PM.png")?;

    let image_tensor =
        image_ops::load_image_as_tensor("images/test/akronim-dist-1-regular-upper-M-img.png")?;
    char_rec_conv_nn::run_prediction(&image_tensor)?;

    Ok(())
}
