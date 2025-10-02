use crate::{audio::*, consts::*, genre::*};
use burn::{
    backend::{Autodiff, Wgpu},
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        conv::{Conv1d, Conv1dConfig},
        loss::CrossEntropyLossConfig,
    },
    tensor::{
        ElementConversion, Int, Tensor, TensorData,
        activation::gelu,
        backend::{AutodiffBackend, Backend},
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

pub type MyAutodiffBackend = Autodiff<Wgpu<f32, i32>>;

#[derive(Clone, Default)]
pub struct AudioBatcher {}

#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, Crop, AudioBatch<B>> for AudioBatcher {
    fn batch(&self, items: Vec<Crop>, device: &B::Device) -> AudioBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.data()))
            .map(|data| Tensor::<B, 1>::from_floats(data, device))
            .map(|tensor| tensor.reshape([1, N_COEFFS * 3, N_SEQS]))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([item.genre().index().elem::<B::IntElem>()], device)
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
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
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
    conv1d: Conv1d<B>,
    mha: MultiHeadAttention<B>,
    linear1: Linear<B>,
    dropout: Dropout,
    classifier: Linear<B>,
}

#[derive(Config, Debug)]
pub struct NetworkConfig {}

impl NetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            conv1d: Conv1dConfig::new(N_COEFFS * 3, 128, 5)
                .with_padding(PaddingConfig1d::Explicit(2))
                .init(device),
            mha: MultiHeadAttentionConfig::new(128, 4)
                .with_quiet_softmax(true)
                .with_dropout(0.3)
                .init(device),
            linear1: LinearConfig::new(128, 128).init(device),
            dropout: DropoutConfig::new(0.3).init(),
            classifier: LinearConfig::new(128, Genre::GENRES_N).init(device),
        }
    }
}

impl<B: Backend> Network<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.conv1d.forward(input);
        let x = x.transpose();
        let mha_input = MhaInput::self_attn(x);
        let x = self.mha.forward(mha_input).context;
        let x = x.mean_dim(1).squeeze(1);
        let x = gelu(self.linear1.forward(x));
        let x = self.dropout.forward(x);
        self.classifier.forward(x)
    }
}
