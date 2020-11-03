use crate::utils::VALUES_COUNT_I64;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    pub fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 512, Default::default());
        let fc2 = nn::linear(vs, 512, VALUES_COUNT_I64, Default::default());
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
