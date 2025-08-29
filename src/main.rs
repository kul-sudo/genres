use bincode::{Decode, Encode, config::standard, decode_from_std_read, encode_into_std_write};
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
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
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
    fs::{File, exists, read_dir, read_to_string},
    io::{BufReader, BufWriter},
    mem::transmute,
    path::{Path, PathBuf},
};
use toml::from_str;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use symphonia::core::probe::Hint;

type MyAutodiffBackend = Autodiff<NdArray>;

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    layer1: BiLstm<B>,
    layer2: Linear<B>,
    layer3: LayerNorm<B>,
    layer4: Dropout,
    layer5: Linear<B>,
    layer6: LayerNorm<B>,
    layer7: Dropout,
    layer8: Linear<B>,
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
            layer2: LinearConfig::new(128, 128).init(device),
            layer3: LayerNormConfig::new(128).init(device),
            layer4: DropoutConfig::new(0.3).init(),
            layer5: LinearConfig::new(128, 128).init(device),
            layer6: LayerNormConfig::new(128).init(device),
            layer7: DropoutConfig::new(0.3).init(),
            layer8: LinearConfig::new(128, self.output_features).init(device),
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
        let (x, _state) = self.layer1.forward(input_3d, None);

        let flat = x.mean_dim(1).squeeze(1);

        let x = relu(self.layer2.forward(flat));
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);
        let x = relu(self.layer5.forward(x));
        let x = self.layer6.forward(x);
        let x = self.layer7.forward(x);

        self.layer8.forward(x)
    }
}

const MAX_SAMPLES_N: usize = (CROP_PAD / FRAME_SIZE) * (N_COEFFS * 3);

const CROP_SECONDS: usize = 5;
const FRAME_SIZE: usize = 2_usize.pow(10);
const N_FILTERS: usize = 40;
const N_COEFFS: usize = 20;

const SAMPLE_RATE: usize = 44100;
const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();

const PART_FOR_TRAINING: f32 = 0.9;

const BATCH_SIZE: usize = 32;

#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, Encode, Decode)]
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
    #[config(default = 500)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

const ITEMS_FILE: &str = "items.bin";

pub fn mfccs_for_audio(path: &PathBuf, all_mfccs: &mut Vec<f64>) {
    let src = File::open(path).unwrap();
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("wav");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) {
        Ok(probed) => probed,
        Err(_) => return,
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

    let mut rng = StdRng::from_os_rng();

    let crop_start =
        (track.codec_params.n_frames.unwrap() as f32 * rng.random_range(0.0..1.0)) as usize;
    let crop_range = crop_start..(crop_start + CROP_PAD);

    let mut n = 0;

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).unwrap();

        let spec = decoded.spec();
        let duration = decoded.capacity();

        let mut buf = SampleBuffer::<f32>::new(duration as u64, *spec);
        buf.copy_interleaved_ref(decoded);

        for &sample in buf.samples().iter() {
            if crop_range.contains(&n) {
                samples.push((sample * i16::MAX as f32) as i16);
            }

            n += 1;
        }

        // if n + samples_n < crop_range.start {
        //     n += samples_n
        // } else if n > crop_range.end {
        //     break 'packet_loop;
        // } else {
        //     for &sample in buf.samples().iter() {
        //         if crop_range.contains(&n) {
        //             samples.push((sample * i16::MAX as f32) as i16);
        //         }
        //
        //         n += 1;
        //     }
        // }
    }

    for chunk in samples.chunks(FRAME_SIZE) {
        if chunk.len() == FRAME_SIZE {
            let mut state =
                Transform::new(track.codec_params.sample_rate.unwrap() as usize, FRAME_SIZE)
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

const ITERATIONS: usize = 100;

pub fn main() {
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
                get_items(&mut items);
                let file = File::create(ITEMS_FILE).unwrap();
                let mut writer = BufWriter::new(file);
                encode_into_std_write(&items, &mut writer, standard()).unwrap();
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

                for _ in 0..ITERATIONS {
                    let data = TensorData::from(all_mfccs.as_slice());
                    let b = Tensor::<MyAutodiffBackend, 1>::from_floats(data, &device);
                    let tensor = b.reshape([1, -1]);

                    let a = model.forward(tensor);

                    let probabilities = softmax(a, 1);

                    let genre_index = probabilities.argmax(1).into_scalar() as u8;
                    let genre = unsafe { transmute::<u8, Genre>(genre_index) };
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
