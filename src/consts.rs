use std::num::NonZeroUsize;

pub const BATCH_SIZE: usize = 16;
pub const INPUT_DIR: &str = "input";
pub const CROPS_N: NonZeroUsize = NonZeroUsize::new(8).unwrap();
pub const TRAINING_SPLIT: f32 = 0.8;
pub const SAMPLE_RATE: usize = 44100;
pub const CACHE_FILE: &str = "cache.bin";
pub const ITERATIONS: usize = 5;
pub const CONFIG_FILE: &str = "config.toml";
pub const ARTIFACT_DIR: &str = "artifact";

pub const FFT_SIZE: usize = 1024;
pub const HOP_SIZE: usize = 256;
pub const N_MELS: usize = 32;
pub const N_FRAMES: usize = 128;

pub const CROP_PAD: usize = (N_FRAMES - 1) * HOP_SIZE + FFT_SIZE;
