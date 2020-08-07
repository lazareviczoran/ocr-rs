#[macro_use]
extern crate lazy_static;
extern crate tch;

mod char_rec_conv_nn;
mod char_rec_nn;
mod image_ops;

use anyhow::Result;

fn main() -> Result<()> {
    // char_rec_conv_nn::run()?;
    char_rec_nn::run()?;
    Ok(())
}
