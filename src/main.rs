#![recursion_limit = "256"]

use burn::{
    backend::{Autodiff, NdArray},
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        lstm::{BiLstm, BiLstmConfig},
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        ElementConversion, Int, Tensor, TensorData,
        activation::{relu, softmax},
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
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::{File, exists, read_dir, read_to_string, write},
    path::Path,
    path::PathBuf,
    sync::OnceLock,
};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use symphonia::core::probe::Hint;

use burn::backend::wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};

// Define our Autodiff backend type
type MyAutodiffBackend = Autodiff<NdArray>;

// -- 1. Two-Layer Network Module Definition --
#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    layer1: BiLstm<B>,
    layer2: Linear<B>,
    layer3: Dropout,
    layer4: Linear<B>,
    layer5: Dropout,
    layer6: Linear<B>,
}

#[derive(Config, Debug)]
pub struct NetworkConfig {
    input_features: usize,
    output_features: usize,
}

impl NetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            layer1: BiLstmConfig::new(self.input_features, 64, true).init(device),
            layer2: LinearConfig::new(128, 64).init(device),
            layer3: DropoutConfig::new(0.5).init(),
            layer4: LinearConfig::new(64, 64).init(device),
            layer5: DropoutConfig::new(0.5).init(),
            layer6: LinearConfig::new(64, self.output_features).init(device),
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
        let (lstm_output, _state) = self.layer1.forward(input_3d, None);

        let flat = lstm_output.mean_dim(1).squeeze(1);

        let x = relu(self.layer2.forward(flat));
        let x = self.layer3.forward(x);
        let x = relu(self.layer4.forward(x));
        let x = self.layer5.forward(x);
        self.layer6.forward(x)
    }
}

static MAX_SAMPLES_N: usize = (CROP_PAD / FRAME_SIZE) * (N_COEFFS * 3);

static CROP_PAD: usize = 2_usize.pow(16);
static FRAME_SIZE: usize = 2_usize.pow(10);
const N_FILTERS: usize = 40;
const N_COEFFS: usize = 20;

const PART_FOR_TRAINING: f32 = 0.9;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
enum Genre {
    Punk,
    RockNRoll,
    Pop,
    Rap,
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
            Genre::Rap => 3,
        }
    }
}

impl From<String> for Genre {
    fn from(string: String) -> Genre {
        match string.as_str() {
            "punk" => Genre::Punk,
            "rocknroll" => Genre::RockNRoll,
            "pop" => Genre::Pop,
            "rap" => Genre::Rap,
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

use serde::Serialize;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AudioItem {
    genre: Genre,
    samples: Vec<f64>,
}

const ARTIFACT_DIR: &str = "artifact";

impl<B: Backend> Batcher<B, AudioItem, AudioBatch<B>> for AudioBatcher {
    fn batch(&self, items: Vec<AudioItem>, device: &B::Device) -> AudioBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.samples.as_slice()))
            .map(|data| Tensor::<B, 1>::from_floats(data, device))
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(i64::from(item.genre.clone())).elem::<B::IntElem>()],
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
    #[config(default = 10000)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

const ITEMS_FILE: &str = "items.json";

static SAMPLE_RATE: OnceLock<u32> = OnceLock::new();

pub fn mfccs_for_audio(path: &PathBuf, all_mfccs: &mut Vec<f64>) {
    let src = File::open(path).unwrap();
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("wav");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .unwrap();

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .unwrap();

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
        let duration = decoded.capacity();

        let _ = SAMPLE_RATE.set(spec.rate);

        let mut buf = SampleBuffer::<f32>::new(duration as u64, *spec);
        buf.copy_interleaved_ref(decoded);
        for &sample in buf.samples().iter() {
            samples.push((sample * i16::MAX as f32) as i16);
        }
    }

    let mut rng = StdRng::from_os_rng();

    let crop_start = (samples.len() as f32 * rng.random_range(0.0..1.0)) as usize;
    let slice = &samples[crop_start..(crop_start + CROP_PAD).min(samples.len())];

    for chunk in slice.chunks(FRAME_SIZE) {
        if chunk.len() == FRAME_SIZE {
            let mut state = Transform::new(*SAMPLE_RATE.get().unwrap() as usize, FRAME_SIZE)
                .nfilters(N_COEFFS, N_FILTERS)
                .normlength(10);

            let mut output = vec![0.0; N_COEFFS * 3];
            state.transform(chunk, &mut output);
            all_mfccs.extend_from_slice(&output);
        }
    }

    all_mfccs.resize(MAX_SAMPLES_N, 0.0);
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

            let mut all_mfccs = Vec::new();
            mfccs_for_audio(&audio_path, &mut all_mfccs);

            items.push(AudioItem {
                genre: genre.clone(),
                samples: all_mfccs,
            });
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

    let mut rng = StdRng::from_os_rng();

    items.shuffle(&mut rng);

    let threshold = (items.len() as f32 * PART_FOR_TRAINING) as usize;
    let for_training = &items[..threshold];
    let for_validation = &items[threshold..];

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(for_training.len() / 7)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(for_training.to_vec()));

    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(for_validation.len() / 2)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(for_validation.to_vec()));

    let a = MetricCheckpointingStrategy::new(
        &AccuracyMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Highest,
        Split::Valid,
    );

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
            CosineAnnealingLrSchedulerConfig::new(config.learning_rate, 50)
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

const ITERATIONS: usize = 50;

pub fn main() {
    let data = read_to_string(CONFIG_FILE).unwrap();

    let config: TomlConfig = toml::from_str(&data).unwrap();

    match config.mode {
        Mode::Train => {
            let mut items = vec![];

            if exists(ITEMS_FILE).unwrap() {
                let data = read_to_string(ITEMS_FILE).unwrap();
                items = serde_json::from_str::<Vec<AudioItem>>(&data).unwrap();
            } else {
                get_items(&mut items);

                let string = serde_json::to_string_pretty(&items).unwrap();
                write(ITEMS_FILE, string).unwrap();
            }

            let device = Default::default();

            train::<MyAutodiffBackend>(
                TrainingConfig::new(
                    NetworkConfig::new(N_COEFFS * 3, Genre::GENRES_N),
                    AdamConfig::new(),
                ),
                &mut items,
                device,
            );
        }
        Mode::Test => {
            let device = Default::default();
            let mut model = NetworkConfig::new(N_COEFFS * 3, Genre::GENRES_N)
                .init::<MyAutodiffBackend>(&device);

            model = model
                .load_file(
                    Path::new("models").join("model"),
                    &CompactRecorder::new(),
                    &device,
                )
                .unwrap();

            for audio in read_dir("test").unwrap() {
                let audio = audio.unwrap();
                let audio_path = audio.path();

                let mut all_mfccs = Vec::new();
                mfccs_for_audio(&audio_path, &mut all_mfccs);

                let mut frequencies = HashMap::with_capacity(ITERATIONS);
                let data = TensorData::from(all_mfccs.as_slice());
                let b = Tensor::<MyAutodiffBackend, 1>::from_floats(data, &device);
                let tensor = b.reshape([1, -1]);

                for _ in 0..ITERATIONS {
                    let a = model.forward(tensor.clone());

                    let probabilities = softmax(a, 1);

                    let genre_index = probabilities.argmax(1).into_scalar() as u8;
                    let genre = unsafe { std::mem::transmute::<u8, Genre>(genre_index) };
                    frequencies
                        .entry(genre)
                        .and_modify(|x| *x += 1)
                        .or_insert(1);
                }

                let most_frequent = frequencies
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .unwrap()
                    .0;

                println!(
                    "{} {:?}",
                    audio_path.file_name().unwrap().to_string_lossy(),
                    most_frequent
                );
            }
        }
    }
}
