mod audio;
mod consts;
mod files;
mod genre;
mod model;

use audio::*;
use bincode::{config::standard, decode_from_std_read, encode_into_std_write};
use burn::{
    backend::{cuda::CudaDevice, ndarray::NdArrayDevice},
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::Module,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
    train::{
        LearnerBuilder, LearningStrategy,
        checkpoint::{ComposedCheckpointingStrategy, MetricCheckpointingStrategy},
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
    io::Write,
    io::{BufReader, BufWriter},
    mem::transmute,
    path::Path,
    sync::LazyLock,
};
use toml::from_str;

#[derive(Debug, Config)]
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
    #[config(default = 8)]
    pub batch_size: usize,
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    items: &mut [Vec<Crop>],
    device: B::Device,
) {
    B::seed(&device, config.seed);

    items.shuffle(&mut rng());

    let threshold = (items.len() as f32 * TRAINING_SPLIT) as usize;
    let (for_training, for_validation) = items.split_at_mut(threshold);

    let steps_per_epoch = for_training.len().div_ceil(config.batch_size);
    let total_steps = steps_per_epoch * config.num_epochs;

    let (mean, std) = compute_global_stats(for_training);

    for split in [&mut for_training[..], &mut for_validation[..]] {
        for crops in split {
            for crop in crops {
                crop.normalize(mean, std);
            }
        }
    }

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .with_file_checkpointer(CompactRecorder::new())
        .with_checkpointing_strategy(
            ComposedCheckpointingStrategy::builder()
                .add(MetricCheckpointingStrategy::new(
                    &AccuracyMetric::<B>::new(),
                    Aggregate::Mean,
                    Direction::Highest,
                    Split::Valid,
                ))
                .build(),
        )
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            CosineAnnealingLrSchedulerConfig::new(config.learning_rate, total_steps)
                .init()
                .unwrap(),
            LearningStrategy::SingleDevice(device.clone()),
        );

    let file = File::create(Path::new(ARTIFACT_DIR).join(STATS_FILE)).unwrap();
    let mut writer = BufWriter::new(file);
    encode_into_std_write((mean, std), &mut writer, standard()).unwrap();
    writer.flush().unwrap();

    let batcher = AudioBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(
            for_training.iter().flatten().cloned().collect::<Vec<_>>(),
        ));

    let dataloader_valid = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(
            for_validation.iter().flatten().cloned().collect::<Vec<_>>(),
        ));

    let result = learner.fit(dataloader_train, dataloader_valid);

    result
        .model
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
pub struct TomlConfig {
    mode: Mode,
}

pub static CONFIG: LazyLock<TomlConfig> = LazyLock::new(|| {
    let data = read_to_string(CONFIG_FILE).unwrap();

    let config: TomlConfig = from_str(&data).unwrap();
    config
});

pub fn main() {
    println!("{} {}", CROP_PAD, CROP_PAD as f32 / SAMPLE_RATE as f32);

    match CONFIG.mode {
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
                writer.flush().unwrap();
            }

            let device = CudaDevice::default();

            train::<TrainingBackend>(
                TrainingConfig::new(
                    NetworkConfig::new(),
                    AdamWConfig::new()
                        .with_cautious_weight_decay(true)
                        .with_weight_decay(1e-2),
                ),
                &mut items,
                device,
            );
        }
        Mode::Test => {
            let path = var(MODEL_VAR).unwrap();

            let device = NdArrayDevice::Cpu;
            let mut model = NetworkConfig::new().init::<TestingBackend>(&device);

            model = model
                .load_file(&path, &CompactRecorder::new(), &device)
                .unwrap();

            let mut entries: Vec<_> = read_dir(TEST_DIR).unwrap().collect();
            entries.shuffle(&mut rng());

            let file = File::open(
                Path::new(&Path::new(&path).components().next().unwrap()).join(STATS_FILE),
            )
            .unwrap();
            let mut reader = BufReader::new(file);
            let (mean, std) = decode_from_std_read(&mut reader, standard()).unwrap();

            for audio in entries {
                let audio = audio.unwrap();
                let audio_path = audio.path();

                let mut frequencies = HashMap::with_capacity(ITERATIONS);

                for _ in 0..ITERATIONS {
                    let mut variations = Crop::prepare(&audio_path).unwrap();

                    for variation in &mut variations {
                        normalize(variation, mean, std);

                        let data = TensorData::from(variation.as_slice());
                        let b = Tensor::<TestingBackend, 1>::from_floats(data, &device);
                        let tensor = b.reshape([1, 1, N_FILTERS, N_FRAMES]);

                        let output = model.forward(tensor);

                        let genre_index = output.argmax(1).into_scalar() as u8;
                        let genre = unsafe { transmute::<u8, Genre>(genre_index) };

                        frequencies
                            .entry(genre)
                            .and_modify(|x| *x += 1)
                            .or_insert(1);
                    }
                }

                let mut freq_vec: Vec<_> = frequencies.into_iter().collect();
                freq_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let top_two = &freq_vec[..TOP_N.min(freq_vec.len())];

                println!(
                    "{} {:?}",
                    audio_path.file_name().unwrap().to_string_lossy(),
                    top_two
                );
            }
        }
    }
}
