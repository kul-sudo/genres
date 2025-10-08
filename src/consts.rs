use std::num::NonZeroUsize;
use std::ops::RangeInclusive;

pub const BATCH_SIZE: usize = 32;
pub const INPUT_DIR: &str = "input";
pub const N_CROPS: NonZeroUsize = NonZeroUsize::new(5).unwrap();
pub const TRAINING_SPLIT: f32 = 0.8;
pub const SAMPLE_RATE: usize = 44100;
pub const CACHE_FILE: &str = "cache.bin";
pub const ITERATIONS: usize = 10;
pub const CONFIG_FILE: &str = "config.toml";
pub const ARTIFACT_DIR: &str = "artifact";

pub const N_FILTERS: usize = 40;
pub const N_COEFFS: usize = 12;

pub const MAX_SAMPLES_N: usize = (CROP_PAD / HOP_LENGTH) * N_COEFFS;
pub const N_SEQS: usize = MAX_SAMPLES_N / N_COEFFS;

// Data augmentation
pub const N_VARIATIONS: usize = 2;
pub const DISTORTION_CHANCE: f32 = 0.5;
pub const DECAY: f32 = 0.3;
pub const DAMPING: f32 = 0.8;
pub const BANDWIDTH: f32 = 0.3;
pub const DRY_RANGE: RangeInclusive<f32> = 0.0..=1.0;
pub const WET_RANGE: RangeInclusive<f32> = 0.0..=1.0;
pub const NOISE_LEVEL: f32 = 0.003;
pub const SEMITONE_VARIANCE: RangeInclusive<f32> = -2.0..=2.0;

// Ideally need to be a power of two
pub const FRAME_SIZE: usize = 2_usize.pow(11);
pub const HOP_LENGTH: usize = 2_usize.pow(9);

const CROP_SECONDS: usize = 4; // More seconds may be read due to the presence "next_power_of_two"
pub const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();
