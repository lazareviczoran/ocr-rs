use super::image_ops;
use super::utils;
use super::DEVICE;
use anyhow::Result;
use log::{debug, info};
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Kind, Tensor};

const MODEL_FILENAME: &str = "char_rec_conv_net.model";

#[derive(Debug)]
pub struct Net {
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
        let fc2 = nn::linear(vs, 512, utils::VALUES_COUNT_I64, Default::default());
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
            .apply_t(&self.conv1, train)
            .max_pool2d_default(2)
            .apply_t(&self.conv2, train)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply_t(&self.fc1, train)
            .relu()
            .dropout_(0.5, train)
            .apply_t(&self.fc2, train)
    }
}

pub fn create_and_train_model() -> Result<Net> {
    let m = image_ops::load_values()?;
    let vs = nn::VarStore::new(*DEVICE);
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

    vs.save(MODEL_FILENAME)?;

    Ok(net)
}

pub fn run_prediction(image_tensor: &Tensor) -> Result<()> {
    let net: Net;
    if !Path::new(MODEL_FILENAME).exists() {
        info!("Started new training process");
        net = create_and_train_model()?;
        info!("Completed the training process")
    } else {
        let mut weights = nn::VarStore::new(*DEVICE);
        net = Net::new(&weights.root());
        weights.load(MODEL_FILENAME)?;
    }

    // recognize character
    let instant = Instant::now();
    let res = net
        .forward_t(&image_tensor, false)
        .softmax(-1, Kind::Double);
    let (predicted_value, probability) = utils::topk(&res, 1)[0];

    debug!(
        "finished classification in {:?} ns, with {} as result with {:3.2}% of certainty",
        instant.elapsed().as_nanos(),
        predicted_value,
        probability * 100.
    );

    Ok(())
}
