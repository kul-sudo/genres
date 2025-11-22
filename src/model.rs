use crate::{audio::*, consts::*, genre::*};
use burn::{
    backend::{Autodiff, Cuda, NdArray},
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{
        Dropout, DropoutConfig, GaussianNoise, GaussianNoiseConfig, InstanceNorm,
        InstanceNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{
        ElementConversion, Int, Tensor, TensorData,
        activation::gelu,
        backend::{AutodiffBackend, Backend},
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

pub type TrainingBackend = Autodiff<Cuda<f32, i32>>;
pub type TestingBackend = NdArray<f32, i32>;

#[derive(Clone, Default)]
pub struct AudioBatcher {}

#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, Crop, AudioBatch<B>> for AudioBatcher {
    fn batch(&self, items: Vec<Crop>, device: &B::Device) -> AudioBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.data()))
            .map(|data| Tensor::<B, 1>::from_floats(data, device))
            .map(|tensor| tensor.reshape([1, 1, N_FILTERS, N_FRAMES]))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(*item.genre() as u32).elem::<B::IntElem>()],
                    device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        AudioBatch { images, targets }
    }
}

impl<B: Backend> Network<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.2))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<AudioBatch<B>, ClassificationOutput<B>> for Network<B> {
    fn step(&self, batch: AudioBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<AudioBatch<B>, ClassificationOutput<B>> for Network<B> {
    fn step(&self, batch: AudioBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    noise1: GaussianNoise,
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    norm1: InstanceNorm<B>,
    dropout1: Dropout,
    conv2: Conv2d<B>,
    pool2: MaxPool2d,
    norm2: InstanceNorm<B>,
    dropout2: Dropout,
    linear1: Linear<B>,
    dropout4: Dropout,
    classifier: Linear<B>,
}

#[derive(Config, Debug)]
pub struct NetworkConfig {}

impl NetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            noise1: GaussianNoiseConfig::new(0.2).init(),
            conv1: Conv2dConfig::new([1, 128], [5, 5])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            norm1: InstanceNormConfig::new(128).init(device),
            dropout1: DropoutConfig::new(0.5).init(),
            conv2: Conv2dConfig::new([128, 128], [5, 5])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            norm2: InstanceNormConfig::new(128).init(device),
            dropout2: DropoutConfig::new(0.5).init(),
            linear1: LinearConfig::new(128, 128).init(device),
            dropout4: DropoutConfig::new(0.5).init(),
            classifier: LinearConfig::new(128, Genre::GENRES_N).init(device),
        }
    }
}

impl<B: Backend> Network<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.noise1.forward(input);
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = gelu(x);
        let x = self.pool1.forward(x);
        let x = self.dropout1.forward(x);

        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        let x = gelu(x);
        let x = self.pool2.forward(x);
        let x = self.dropout2.forward(x);

        let x = x.mean_dim(2).mean_dim(3).squeeze_dims(&[2, 3]);
        let x = gelu(self.linear1.forward(x));
        let x = self.dropout4.forward(x);
        self.classifier.forward(x)
    }
}
