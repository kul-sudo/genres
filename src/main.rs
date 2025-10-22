#![recursion_limit = "256"]
mod audio;
mod consts;
mod files;
mod genre;
mod model;

use audio::*;
use bincode::{config::standard, decode_from_std_read, encode_into_std_write};
use burn::{
    backend::wgpu::WgpuDevice,
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::Module,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
    train::{
        LearnerBuilder,
        checkpoint::MetricCheckpointingStrategy,
        metric::{
            AccuracyMetric, CpuTemperature, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use consts::*;
use files::*;
use genre::*;
use model::*;
use rand::{rng, seq::SliceRandom};
use serde::Deserialize;
use std::{
    collections::HashMap,
    env::var,
    fs::{File, exists, read_dir, read_to_string},
    io::{BufReader, BufWriter},
    mem::transmute,
    path::Path,
};
use toml::from_str;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: NetworkConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 4000)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    items: &mut [Vec<Crop>],
    device: B::Device,
) {
    B::seed(config.seed);

    let batcher = AudioBatcher::default();

    items.shuffle(&mut rng());

    let threshold = (items.len() as f32 * TRAINING_SPLIT) as usize;
    let (for_training, for_validation) = items.split_at(threshold);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(BATCH_SIZE)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(
            for_training.iter().flatten().cloned().collect::<Vec<_>>(),
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(BATCH_SIZE)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(
            for_validation.iter().flatten().cloned().collect::<Vec<_>>(),
        ));

    let strategy = MetricCheckpointingStrategy::new(
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
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .with_file_checkpointer(CompactRecorder::new())
        .with_checkpointing_strategy(strategy)
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
    // BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include/aubio" cargo run --release
    println!(
        "{} {} {}",
        MAX_SAMPLES_N,
        CROP_PAD,
        CROP_PAD as f32 / SAMPLE_RATE as f32
    );
    let data = read_to_string(CONFIG_FILE).unwrap();

    let config: TomlConfig = from_str(&data).unwrap();

    match config.mode {
        Mode::Train => {
            let mut items;

            if exists(CACHE_FILE).unwrap() {
                let file = File::open(CACHE_FILE).unwrap();
                let mut reader = BufReader::new(file);
                items = decode_from_std_read(&mut reader, standard()).unwrap();
            } else {
                items = files_init();
                let file = File::create(CACHE_FILE).unwrap();
                let mut writer = BufWriter::new(file);
                encode_into_std_write(&items, &mut writer, standard()).unwrap();
            }

            let device = WgpuDevice::default();

            train::<MyAutodiffBackend>(
                TrainingConfig::new(
                    NetworkConfig::new(),
                    AdamWConfig::new().with_weight_decay(1e-4),
                ),
                &mut items,
                device,
            );
        }
        Mode::Test => {
            let path = var("MODEL").unwrap();

            let device = WgpuDevice::default();
            let mut model = NetworkConfig::new().init::<MyAutodiffBackend>(&device);

            model = model
                .load_file(path, &CompactRecorder::new(), &device)
                .unwrap();

            for audio in read_dir("test").unwrap() {
                let audio = audio.unwrap();
                let audio_path = audio.path();

                let mut frequencies = HashMap::with_capacity(ITERATIONS);

                for _ in 0..ITERATIONS {
                    let variations = Crop::prepare(&audio_path, true).unwrap();

                    for variation in variations {
                        let data = TensorData::from(variation.as_slice());
                        let b = Tensor::<MyAutodiffBackend, 1>::from_floats(data, &device);
                        let tensor = b.reshape([1, N_COEFFS, N_SEQS]);

                        let a = model.forward(tensor);

                        let genre_index = a.argmax(1).into_scalar() as u8;
                        let genre = unsafe { transmute::<u8, Genre>(genre_index) };
                        frequencies
                            .entry(genre)
                            .and_modify(|x| *x += 1)
                            .or_insert(1);
                    }
                }

                let mut freq_vec: Vec<(Genre, u32)> = frequencies.into_iter().collect();
                freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
                let top_two = &freq_vec[..freq_vec.len().min(2)];

                println!(
                    "{} {:?}",
                    audio_path.file_name().unwrap().to_string_lossy(),
                    top_two
                );
            }
        }
    }
}
