use crate::lazy_init::*;
use std::sync::LazyLock;

pub struct MfccParams {
    std: f64,
    mean: f64,
}

impl MfccParams {
    pub fn pair(&self) -> (f64, f64) {
        (self.std, self.mean)
    }
}

pub static MFCC_PARAMS: LazyLock<MfccParams> = LazyLock::new(|| {
    let mut all_mfccs = vec![];

    for paths in FILES.values() {
        for audio in paths {
            if let Some(mfcc) = audio.data() {
                all_mfccs.extend_from_slice(mfcc.data());
            }
        }
    }

    let mean = all_mfccs.iter().sum::<f64>() / all_mfccs.len() as f64;
    let std: f64 =
        (all_mfccs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_mfccs.len() as f64).sqrt();

    MfccParams { std, mean }
});
