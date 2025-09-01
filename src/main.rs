#![recursion_limit = "256"]

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
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
        gru::{Gru, GruConfig},
        loss::CrossEntropyLossConfig,
        pool::{MaxPool1d, MaxPool1dConfig},
    },
    optim::AdamConfig,
    optim::decay::WeightDecayConfig,
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
use mfcc::Transform;
use rand::{
    Rng, rng,
    seq::{IndexedRandom, SliceRandom},
};
use serde::Deserialize;
use std::{
    fs::{File, create_dir, exists, read_dir, read_to_string},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    sync::LazyLock,
    time::{SystemTime, UNIX_EPOCH},
};
use toml::from_str;

use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

type MyAutodiffBackend = Autodiff<Wgpu<f32, i32>>;

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    conv1: Conv1d<B>,
    maxpool1: MaxPool1d,
    dropout1: Dropout,
    gru1: Gru<B>,
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
            conv1: Conv1dConfig::new(self.input_features, 64, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            maxpool1: MaxPool1dConfig::new(2).init(),
            dropout1: DropoutConfig::new(0.3).init(),
            gru1: GruConfig::new(64, 128, true).init(device),
            linear1: LinearConfig::new(128, 128).init(device),
            linear2: LinearConfig::new(128, self.output_features).init(device),
        }
    }
}

const INPUT_SIZE: usize = N_COEFFS * 3;

impl<B: Backend> Network<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        debug_assert_eq!(
            input.dims()[1],
            N_COEFFS * 3 * (MAX_SAMPLES_N / (N_COEFFS * 3)),
            "Input size mismatch: expected {}, got {}",
            N_COEFFS * 3 * (MAX_SAMPLES_N / (N_COEFFS * 3)),
            input.dims()[1]
        );

        let dims = input.dims();
        let mut x = input.reshape([dims[0], INPUT_SIZE, CROP_PAD / FRAME_SIZE]);

        x = gelu(self.conv1.forward(x));
        x = self.maxpool1.forward(x);
        x = self.dropout1.forward(x);

        let mut x = x.transpose();
        x = self.gru1.forward(x, None);

        let mut x = x.mean_dim(1).squeeze(1);
        x = gelu(self.linear1.forward(x));
        x = self.linear2.forward(x);

        x
    }
}

const MAX_SAMPLES_N: usize = (CROP_PAD / FRAME_SIZE) * (N_COEFFS * 3);

const CROP_SECONDS: usize = 4;
const FRAME_SIZE: usize = 2_usize.pow(10);
const N_FILTERS: usize = 40;
const N_COEFFS: usize = 20;

const SAMPLE_RATE: usize = 44100;
const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();

const PART_FOR_TRAINING: f32 = 0.9;

const BATCH_SIZE: usize = 32;

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
enum Genre {
    Punk,
    RockNRoll,
    Pop,
    Electronic,
}

impl Genre {
    const GENRES_N: usize = 4;
}

impl From<Genre> for i64 {
    fn from(genre: Genre) -> i64 {
        match genre {
            Genre::Punk => 0,
            Genre::RockNRoll => 1,
            Genre::Pop => 2,
            Genre::Electronic => 3,
        }
    }
}

impl From<String> for Genre {
    fn from(string: String) -> Genre {
        match string.as_str() {
            "punk" => Genre::Punk,
            "rocknroll" => Genre::RockNRoll,
            "pop" => Genre::Pop,
            "electronic" => Genre::Electronic,
            _ => panic!(),
        }
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
    crops: Vec<PathBuf>,
}

const ARTIFACT_DIR: &str = "artifact";

impl<B: Backend> Batcher<B, AudioItem, AudioBatch<B>> for AudioBatcher {
    fn batch(&self, items: Vec<AudioItem>, device: &B::Device) -> AudioBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                let crop = item.crops.choose(&mut rng()).unwrap();
                let file = File::open(crop).unwrap();
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

static ITEMS_FILE: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::from(ITEMS_DIR).join("items.bin"));
const ITEMS_DIR: &str = "items";
const CROPS: usize = 4;

pub fn audio_crops(path: &PathBuf) -> Option<Vec<PathBuf>> {
    let src = File::open(path).unwrap();
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("wav");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) {
        Ok(probed) => probed,
        Err(_) => return None,
    };

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .unwrap()
        .clone();

    let dec_opts: DecoderOptions = Default::default();

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .unwrap();

    let track_id = track.id;

    let mut samples = vec![];

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).unwrap();

        let spec = decoded.spec();
        assert_eq!(spec.rate as usize, SAMPLE_RATE);

        let channels_n = spec.channels.count();

        let duration = decoded.capacity();

        let mut buf = SampleBuffer::<f32>::new(duration as u64, *spec);
        buf.copy_interleaved_ref(decoded);

        let buf_samples = buf.samples();

        for sample in buf_samples.chunks_exact(channels_n) {
            let mono = sample.iter().sum::<f32>() / sample.len() as f32;
            samples.push((mono * i16::MAX as f32) as i16);
        }
    }

    let mut state = Transform::new(SAMPLE_RATE, FRAME_SIZE).nfilters(N_COEFFS, N_FILTERS);

    let mut crops = Vec::with_capacity(CROPS);

    for _ in 0..CROPS {
        let mut all_mfccs = Vec::with_capacity(MAX_SAMPLES_N);
        let crop_start = rng().random_range(0..samples.len() - CROP_PAD) as usize;
        let crop_range = crop_start..(crop_start + CROP_PAD);
        for chunk in samples[crop_range].chunks_exact(FRAME_SIZE) {
            let mut output = vec![0.0; N_COEFFS * 3];
            state.transform(chunk, &mut output);
            all_mfccs.extend_from_slice(&output);
        }

        assert_eq!(all_mfccs.len(), MAX_SAMPLES_N);

        let min_mfcc = -100.0;
        let max_mfcc = 100.0;
        let range = max_mfcc - min_mfcc;

        for val in all_mfccs.iter_mut() {
            *val = 2.0 * (*val - min_mfcc) / range - 1.0;
            // Clamping (optional)
            *val = val.clamp(-1.0, 1.0);
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from(ITEMS_DIR).join(format!("{now}.bin"));
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        encode_into_std_write(&all_mfccs, &mut writer, standard()).unwrap();

        crops.push(path)
    }

    Some(crops)
}

pub fn get_items(items: &mut Vec<AudioItem>) {
    for entry in read_dir("genres").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap();
        let genre = Genre::from(name.to_string_lossy().to_string());

        for audio in read_dir(&path).unwrap() {
            let audio = audio.unwrap();
            let audio_path = audio.path();

            if let Some(crops) = audio_crops(&audio_path) {
                items.push(AudioItem { genre, crops });
            }
        }
    }
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
    let for_training = &items[..threshold];
    let for_validation = &items[threshold..];

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

const CONFIG_FILE: &str = "config.toml";

#[derive(Deserialize)]
enum Mode {
    Train,
    Test,
}

#[derive(Deserialize)]
struct TomlConfig {
    mode: Mode,
}

const ITERATIONS: usize = 1;

pub fn main() {
    if !exists(ITEMS_DIR).unwrap() {
        create_dir(ITEMS_DIR).unwrap();
    }

    let data = read_to_string(CONFIG_FILE).unwrap();

    let config: TomlConfig = from_str(&data).unwrap();

    match config.mode {
        Mode::Train => {
            let mut items = vec![];

            if exists(&*ITEMS_FILE).unwrap() {
                let file = File::open(&*ITEMS_FILE).unwrap();
                let mut reader = BufReader::new(file);
                items = decode_from_std_read(&mut reader, standard()).unwrap();
            } else {
                get_items(&mut items);
                let file = File::create(&*ITEMS_FILE).unwrap();
                let mut writer = BufWriter::new(file);
                encode_into_std_write(&items, &mut writer, standard()).unwrap();
            }

            let device = WgpuDevice::default();

            train::<MyAutodiffBackend>(
                TrainingConfig::new(
                    NetworkConfig::new(INPUT_SIZE, Genre::GENRES_N),
                    AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(0.02))),
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
