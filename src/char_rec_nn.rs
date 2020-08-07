use super::image_ops;
use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES_L1: i64 = 700;
const HIDDEN_NODES_L2: i64 = 500;
const LABELS: i64 = image_ops::VALUES_COUNT as i64;

fn net(vs: &nn::Path) -> impl Module {
  nn::seq()
    .add(nn::linear(
      vs / "layer1",
      IMAGE_DIM,
      HIDDEN_NODES_L1,
      Default::default(),
    ))
    .add_fn(|xs| xs.relu())
    .add(nn::linear(
      vs / "layer2",
      HIDDEN_NODES_L1,
      HIDDEN_NODES_L2,
      Default::default(),
    ))
    .add_fn(|xs| xs.relu())
    .add(nn::linear(vs, HIDDEN_NODES_L2, LABELS, Default::default()))
}

pub fn run() -> Result<()> {
  let m = image_ops::load_values()?;
  let vs = nn::VarStore::new(Device::Cpu);
  let net = net(&vs.root());
  let mut opt = nn::Adam::default().build(&vs, 0.01)?;
  for epoch in 1..=100 {
    let loss = net
      .forward(&m.train_images)
      .cross_entropy_for_logits(&m.train_labels);
    opt.backward_step(&loss);
    let test_accuracy = net
      .forward(&m.test_images)
      .accuracy_for_logits(&m.test_labels);
    println!(
      "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
      epoch,
      f64::from(&loss),
      100. * f64::from(&test_accuracy),
    );
  }
  Ok(())
}
