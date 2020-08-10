use super::image_ops;
use anyhow::Result;
use log::info;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

#[derive(Debug)]
struct Net {
  conv1: nn::Conv2D,
  conv2: nn::Conv2D,
  fc1: nn::Linear,
  fc2: nn::Linear,
}

impl Net {
  fn new(vs: &nn::Path) -> Net {
    let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
    let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
    let fc1 = nn::linear(vs, 1024, 512, Default::default());
    let fc2 = nn::linear(vs, 512, image_ops::VALUES_COUNT as i64, Default::default());
    Net {
      conv1,
      conv2,
      fc1,
      fc2,
    }
  }
}

impl nn::ModuleT for Net {
  fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
    xs.view([-1, 1, 28, 28])
      .apply(&self.conv1)
      .max_pool2d_default(2)
      .apply(&self.conv2)
      .max_pool2d_default(2)
      .view([-1, 1024])
      .apply(&self.fc1)
      .relu()
      .dropout_(0.5, train)
      .apply(&self.fc2)
  }
}

pub fn run() -> Result<()> {
  let m = image_ops::load_values()?;
  let vs = nn::VarStore::new(Device::cuda_if_available());
  let net = Net::new(&vs.root());
  let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
  for epoch in 1..=100 {
    for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
      let loss = net
        .forward_t(&bimages, true)
        .cross_entropy_for_logits(&blabels);
      opt.backward_step(&loss);
    }
    let test_accuracy =
      net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
    info!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
  }
  Ok(())
}
