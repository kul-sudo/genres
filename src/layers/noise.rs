use burn::{
    config::Config,
    module::Module,
    tensor::{
        backend::Backend,
        {Distribution, Tensor},
    },
};

#[derive(Config, Debug)]
pub struct NoiseConfig {
    pub std: f64,
}

#[derive(Module, Clone, Debug)]
pub struct Noise {
    pub std: f64,
}

impl NoiseConfig {
    pub fn init(&self) -> Noise {
        if self.std.is_sign_negative() {
            panic!(
                "Standard deviation is required to be non-negative, but got {}",
                self.std
            );
        }
        Noise { std: self.std }
    }
}

impl Noise {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if B::ad_enabled() && self.std != 0.0 {
            let noise = Tensor::random(
                input.shape(),
                Distribution::Normal(0.0, self.std),
                &input.device(),
            );
            input + noise
        } else {
            input
        }
    }
}
