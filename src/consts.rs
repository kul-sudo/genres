pub const ITERATIONS: usize = 1;
pub const INPUT_SIZE: usize = N_COEFFS * 3;
pub const MAX_SAMPLES_N: usize = (CROP_PAD / FRAME_SIZE) * (N_COEFFS * 3);
const CROP_SECONDS: usize = 4; // More second may be read due to the presence "next_power_of_two"
pub const FRAME_SIZE: usize = 2_usize.pow(10); // Ideally needs to be a power of two
pub const N_FILTERS: usize = 40;
pub const N_COEFFS: usize = 20;
pub const SAMPLE_RATE: usize = 44100; // Sample rate can be fixed, since every file that's thrown
// in is required to have it
pub const CROP_PAD: usize = (SAMPLE_RATE * CROP_SECONDS).next_power_of_two();
pub const PART_FOR_TRAINING: f32 = 0.9; // What part of the entire dataset is used for training
// (the rest goes into validation)
pub const BATCH_SIZE: usize = 32;

pub const CONFIG_FILE: &str = "config.toml";
pub const ARTIFACT_DIR: &str = "artifact";
pub const ITEMS_FILE: &str = "items.bin";
pub const ITEMS_DIR: &str = "items";
pub const GENRES_DIR: &str = "genres";
