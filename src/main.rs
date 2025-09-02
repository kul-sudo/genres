#![recursion_limit = "256"]
mod audio;
mod consts;
mod genre;
mod lazy_init;

use bincode::{Decode, Encode, config::standard, decode_from_std_read, encode_into_std_write};
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig,
        gru::{Gru, GruConfig},
        loss::CrossEntropyLossConfig,
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        ElementConversion, Int, Tensor, TensorData,
        activation::gelu,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        checkpoint::MetricCheckpointingStrategy,
        metric::{
            AccuracyMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use consts::*;
use genre::*;
use lazy_init::*;
use rand::{rng, seq::SliceRandom};
use serde::Deserialize;
use std::{
    fs::{File, create_dir, exists, read_to_string},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};
use toml::from_str;

type MyAutodiffBackend = Autodiff<Wgpu<f32, i32>>;

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    gru1: Gru<B>,
    dropout1: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct NetworkConfig {
    input_features: usize,
    output_features: usize,
}

impl NetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            gru1: GruConfig::new(self.input_features, 64, true).init(device),
            dropout1: DropoutConfig::new(0.3).init(),
            linear1: LinearConfig::new(64, 64).init(device),
            linear2: LinearConfig::new(64, self.output_features).init(device),
        }
    }
}

impl<B: Backend> Network<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let input_size = N_COEFFS * 3;

        let seq_len = MAX_SAMPLES_N / input_size;

        let input_3d = input
            .clone()
            .reshape([input.dims()[0], seq_len, input_size]);

        let mut x = self.gru1.forward(input_3d, None);
        x = self.dropout1.forward(x);

        let mut x = x.max_dim(1).squeeze(1);
        x = gelu(self.linear1.forward(x));
        x = self.linear2.forward(x);

        x
    }
}

#[derive(Clone, Default)]
pub struct AudioBatcher {}

#[derive(Clone, Debug)]
pub struct AudioBatch<B: Backend> {
    pub images: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug, Encode, Decode)]
pub struct AudioItem {
    genre: Genre,
    crop: PathBuf,
}

impl AudioItem {
    pub fn get_items(items: &mut Vec<AudioItem>) {
        for (genre, paths) in &*FILES {
            for (_, data) in paths {
                if let Some(mfcc) = data {
                    let mut b = mfcc.clone();
                    b.normalize();
                    let crop = b.save();
                    items.push(AudioItem {
                        genre: *genre,
                        crop,
                    });
                }
            }
        }
    }
}

impl<B: Backend> Batcher<B, AudioItem, AudioBatch<B>> for AudioBatcher {
    fn batch(&self, items: Vec<AudioItem>, device: &B::Device) -> AudioBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                let file = File::open(item.crop.clone()).unwrap();
                let mut reader = BufReader::new(file);
                let items: Vec<f64> = decode_from_std_read(&mut reader, standard()).unwrap();

                TensorData::from(items.as_slice())
            })
            .map(|data| Tensor::<B, 1>::from_floats(data, device))
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(i64::from(item.genre)).elem::<B::IntElem>()],
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
        images: Tensor<B, 2>,
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

#[derive(Config)]
pub struct TrainingConfig {
    pub model: NetworkConfig,
    pub optimizer: AdamConfig,
    #[config(default = 4000)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    items: &mut [AudioItem],
    device: B::Device,
) {
    B::seed(config.seed);

    let batcher = AudioBatcher::default();

    items.shuffle(&mut rng());

    let threshold = (items.len() as f32 * PART_FOR_TRAINING) as usize;
    let (for_training, for_validation) = items.split_at(threshold);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(BATCH_SIZE)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(for_training.to_vec()));

    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(BATCH_SIZE)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(for_validation.to_vec()));

    let a = MetricCheckpointingStrategy::new(
        &AccuracyMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Highest,
        Split::Valid,
    );

    let steps_per_epoch = for_training.len().div_ceil(BATCH_SIZE);
    let total_steps = steps_per_epoch * config.num_epochs;

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .with_checkpointing_strategy(a)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            CosineAnnealingLrSchedulerConfig::new(config.learning_rate, total_steps)
                .init()
                .unwrap(),
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(
            Path::new(ARTIFACT_DIR).join("model"),
            &CompactRecorder::new(),
        )
        .expect("Trained model should be saved successfully");
}

#[derive(Deserialize)]
enum Mode {
    Train,
    Test,
}

#[derive(Deserialize)]
struct TomlConfig {
    mode: Mode,
}

pub fn main() {
    if !exists(ITEMS_DIR).unwrap() {
        create_dir(ITEMS_DIR).unwrap();
    }

    let data = read_to_string(CONFIG_FILE).unwrap();

    let config: TomlConfig = from_str(&data).unwrap();

    match config.mode {
        Mode::Train => {
            let mut items = vec![];

            if exists(ITEMS_FILE).unwrap() {
                let file = File::open(ITEMS_FILE).unwrap();
                let mut reader = BufReader::new(file);
                items = decode_from_std_read(&mut reader, standard()).unwrap();
            } else {
                AudioItem::get_items(&mut items);
                let file = File::create(ITEMS_FILE).unwrap();
                let mut writer = BufWriter::new(file);
                encode_into_std_write(&items, &mut writer, standard()).unwrap();
            }

            let device = WgpuDevice::default();

            train::<MyAutodiffBackend>(
                TrainingConfig::new(
                    NetworkConfig::new(INPUT_SIZE, Genre::GENRES_N),
                    AdamConfig::new(),
                ),
                &mut items,
                device,
            );
        }
        Mode::Test => {
            // let device = WgpuDevice::default();
            // let mut model = NetworkConfig::new(N_COEFFS * 3, Genre::GENRES_N)
            //     .init::<MyAutodiffBackend>(&device);
            //
            // model = model
            //     .load_file(
            //         Path::new("models").join("model"),
            //         &CompactRecorder::new(),
            //         &device,
            //     )
            //     .unwrap();
            //
            // for audio in read_dir("test").unwrap() {
            //     let audio = audio.unwrap();
            //     let audio_path = audio.path();
            //
            //     let mut all_mfccs = Vec::new();
            //     mfccs_for_audio(&audio_path, &mut all_mfccs);
            //
            //     let mut frequencies = HashMap::with_capacity(ITERATIONS);
            //
            //     for _ in 0..ITERATIONS {
            //         let data = TensorData::from(all_mfccs.as_slice());
            //         let b = Tensor::<MyAutodiffBackend, 1>::from_floats(data, &device);
            //         let tensor = b.reshape([1, -1]);
            //
            //         let a = model.forward(tensor);
            //
            //         let probabilities = softmax(a, 1);
            //
            //         let genre_index = probabilities.argmax(1).into_scalar() as u8;
            //         let genre = unsafe { transmute::<u8, Genre>(genre_index) };
            //         frequencies
            //             .entry(genre)
            //             .and_modify(|x| *x += 1)
            //             .or_insert(1);
            //     }
            //
            //     let most_frequent = frequencies
            //         .into_iter()
            //         .max_by_key(|&(_, count)| count)
            //         .unwrap()
            //         .0;
            //
            //     println!(
            //         "{} {:?}",
            //         audio_path.file_name().unwrap().to_string_lossy(),
            //         most_frequent
            //     );
            // }
        }
    }
}
