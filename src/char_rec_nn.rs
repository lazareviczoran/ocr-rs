use super::image_ops;
use super::utils;
use anyhow::Result;
use log::{debug, info};
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES_L1: i64 = 700;
const HIDDEN_NODES_L2: i64 = 500;
const LABELS: i64 = utils::VALUES_COUNT_I64;
const MODEL_FILENAME: &str = "char_recognition_mlp.model";

pub fn create_net(vs: &nn::Path) -> Box<dyn Module> {
  Box::new(
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
      .add(nn::linear(vs, HIDDEN_NODES_L2, LABELS, Default::default())),
  )
}

pub fn train_model() -> Result<Box<dyn Module>> {
  let m = image_ops::load_values()?;
  let vs = nn::VarStore::new(Device::cuda_if_available());
  let net = create_net(&vs.root());
  let mut opt = nn::Adam::default().build(&vs, 0.01)?;
  for epoch in 1..=100 {
    let loss = net
      .forward(&m.train_images)
      .cross_entropy_for_logits(&m.train_labels);
    opt.backward_step(&loss);
    let test_accuracy = net
      .forward(&m.test_images)
      .accuracy_for_logits(&m.test_labels);
    info!(
      "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
      epoch,
      f64::from(&loss),
      100. * f64::from(&test_accuracy),
    );
  }
  vs.save(MODEL_FILENAME)?;

  Ok(net)
}

pub fn run_prediction(image_tensor: &Tensor) -> Result<()> {
  let net: Box<dyn Module>;
  if !Path::new(MODEL_FILENAME).exists() {
    info!("Started new training process");
    net = train_model()?;
    info!("Completed the training process")
  } else {
    let mut weights = nn::VarStore::new(Device::cuda_if_available());
    net = create_net(&weights.root());
    weights.load(MODEL_FILENAME)?;
  }

  // recognize character
  let instant = Instant::now();
  let res = net.forward(&image_tensor).softmax(-1, Kind::Float);
  let (predicted_value, probability) = utils::topk(&res, 1)[0];

  debug!(
    "finished classification in {:?} ns, with {} as result with {:3.2}% of certainty",
    instant.elapsed().as_nanos(),
    predicted_value,
    probability * 100.
  );

  Ok(())
}
