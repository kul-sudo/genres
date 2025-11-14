use std::{num::NonZeroUsize, ops::RangeInclusive};

pub const INPUT_DIR: &str = "input";
pub const N_CROPS: NonZeroUsize = NonZeroUsize::new(2).unwrap();
pub const TRAINING_SPLIT: f32 = 0.8;
pub const SAMPLE_RATE: usize = 44100;
pub const CACHE_FILE: &str = "cache.bin";
pub const ITERATIONS: usize = 40;
pub const CONFIG_FILE: &str = "config.toml";
pub const STATS_FILE: &str = "stats.bin";
pub const ARTIFACT_DIR: &str = "artifact";

pub const N_FILTERS: usize = 40;

// Data augmentation
// Training
pub const SIGNALSMITH_CHANCE_TRAINING: f32 = 0.5;
pub const LOFI_CHANCE_TRAINING: f32 = 0.1;

// Testing
pub const SIGNALSMITH_CHANCE_TESTING: f32 = 1.0;
pub const LOFI_CHANCE_TESTING: f32 = 0.1;

pub const PITCH_SHIFT_RANGE: RangeInclusive<f32> = -2.0..=2.0;
pub const SPEED_STRETCH_RANGE: RangeInclusive<f32> = 0.6..=1.5;
pub const NOISE_RANGE: RangeInclusive<f32> = -0.002..=0.002;
pub const HIGH_PASS_RANGE: RangeInclusive<f32> = 80.0..=120.0;
pub const LOW_PASS_RANGE: RangeInclusive<f32> = 2800.0..=3200.0;

// Ideally need to be a power of two
pub const FRAME_SIZE: usize = 2_usize.pow(11);
pub const HOP_LENGTH: usize = 2_usize.pow(10);
pub const N_FRAMES: usize = (CROP_PAD - FRAME_SIZE) / HOP_LENGTH + 1;

const CROP_SECONDS: usize = 5; // More seconds may be read due to the presence "next_power_of_two"
pub const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();
