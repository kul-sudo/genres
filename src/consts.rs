use std::{num::NonZeroUsize, ops::RangeInclusive};

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
pub const N_COEFFS: usize = 12;

pub const MAX_SAMPLES_N: usize = (CROP_PAD / HOP_LENGTH) * N_COEFFS;
pub const N_SEQS: usize = MAX_SAMPLES_N / N_COEFFS;

// Data augmentation
// Training
pub const LOFI_CHANCE_TRAINING: f32 = 0.02;
pub const PITCH_SHIFT_CHANCE_TRAINING: f32 = 0.02;

// Testing
pub const LOFI_CHANCE_TESTING: f32 = 0.5;
pub const PITCH_SHIFT_CHANCE_TESTING: f32 = 0.5;

pub const NOISE_LEVEL: f32 = 0.002;
pub const HIGH_PASS: f32 = 100.0;
pub const LOW_PASS: f32 = 3000.0;
pub const CLAMP_LEVEL: f32 = 0.2;
pub const PITCH_SHIFT_RANGE: RangeInclusive<f32> = -4.0..=4.0;

// Ideally need to be a power of two
pub const FRAME_SIZE: usize = 2_usize.pow(11);
pub const HOP_LENGTH: usize = 2_usize.pow(9);

const CROP_SECONDS: usize = 4; // More seconds may be read due to the presence "next_power_of_two"
pub const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();
