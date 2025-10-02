use std::num::NonZeroUsize;

pub const BATCH_SIZE: usize = 32;
pub const INPUT_DIR: &str = "input";
pub const N_CROPS: NonZeroUsize = NonZeroUsize::new(5).unwrap();
pub const TRAINING_SPLIT: f32 = 0.8;
pub const SAMPLE_RATE: usize = 44100;
pub const CACHE_FILE: &str = "cache.bin";
pub const ITERATIONS: usize = 5;
pub const CONFIG_FILE: &str = "config.toml";
pub const ARTIFACT_DIR: &str = "artifact";

pub const N_FILTERS: usize = 40;
pub const N_COEFFS: usize = 20;

pub const MAX_SAMPLES_N: usize = (CROP_PAD / FRAME_SIZE) * N_COEFFS * 3;
pub const N_SEQS: usize = MAX_SAMPLES_N / (N_COEFFS * 3);
const CROP_SECONDS: usize = 4; // More seconds may be read due to the presence "next_power_of_two"
pub const FRAME_SIZE: usize = 2_usize.pow(10); // Ideally needs to be a power of two
pub const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();
