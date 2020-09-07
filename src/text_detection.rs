use anyhow::Result;
use tch::vision::{imagenet, resnet};
use tch::{nn, nn::OptimizerConfig, CModule, Device, Kind, Tensor};

pub fn find_text_bounding_boxes(image_path: &str) -> Result<()> {
    let image_tensor = imagenet::load_image(image_path)?;

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut model = resnet::resnet18(&vs.root(), 1000);
    // let output = image_tensor.unsqueeze(0).apply(&model);

    // let mut weights = nn::VarStore::new(Device::cuda_if_available());
    // let net = resnet::resnet50_no_final_layer(&weights.root());
    // weights.load("tt_e2e_attn_R_50.pth")?;

    // let output = net.forward_t(&image_tensor, false);
    // println!("{:?}", output.to_string(80));

    Ok(())
}
