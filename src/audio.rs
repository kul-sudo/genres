use crate::consts::*;
use crate::lazy_init::*;
use bincode::{config::standard, encode_into_std_write};
use mfcc::Transform;
use rand::{Rng, rng};
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::LazyLock,
    time::{SystemTime, UNIX_EPOCH},
};

use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

pub struct MfccParams {
    std: f64,
    mean: f64,
}

pub static MFCC_PARAMS: LazyLock<MfccParams> = LazyLock::new(|| {
    let mut all_mfccs = vec![];

    for paths in FILES.values() {
        for (_, data) in paths {
            if let Some(mfcc) = data {
                all_mfccs.extend_from_slice(mfcc.data());
            }
        }
    }

    let mean = all_mfccs.iter().sum::<f64>() / all_mfccs.len() as f64;
    let std: f64 =
        (all_mfccs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_mfccs.len() as f64).sqrt();

    MfccParams { std, mean }
});

pub enum MfccSource {
    Path(PathBuf),
    Samples(Vec<i16>),
}

#[derive(Clone)]
pub struct MfccData {
    data: Vec<f64>,
}

impl MfccData {
    pub fn normalize(&mut self) {
        for val in self.data.iter_mut() {
            *val = (*val - MFCC_PARAMS.mean) / (MFCC_PARAMS.std + 1e-8);
        }
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn save(&self) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = Path::new(ITEMS_DIR).join(format!("{now}.bin"));
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        encode_into_std_write(&self.data, &mut writer, standard()).unwrap();

        path
    }

    pub fn new(source: MfccSource) -> Option<MfccData> {
        let samples = match source {
            MfccSource::Path(path) => {
                let src = File::open(path).unwrap();
                let mss = MediaSourceStream::new(Box::new(src), Default::default());

                let mut hint = Hint::new();
                hint.with_extension("wav");

                let meta_opts: MetadataOptions = Default::default();
                let fmt_opts: FormatOptions = Default::default();

                let probed = match symphonia::default::get_probe()
                    .format(&hint, mss, &fmt_opts, &meta_opts)
                {
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

                samples
            }
            MfccSource::Samples(samples) => samples,
        };

        let mut state = Transform::new(SAMPLE_RATE, FRAME_SIZE).nfilters(N_COEFFS, N_FILTERS);

        let mut all_mfccs = Vec::with_capacity(MAX_SAMPLES_N);
        let crop_start = rng().random_range(0..samples.len() - CROP_PAD) as usize;
        let crop_range = crop_start..(crop_start + CROP_PAD);
        for chunk in samples[crop_range].chunks_exact(FRAME_SIZE) {
            let mut output = vec![0.0; N_COEFFS * 3];
            state.transform(chunk, &mut output);
            all_mfccs.extend_from_slice(&output);
        }

        assert_eq!(all_mfccs.len(), MAX_SAMPLES_N);

        Some(MfccData { data: all_mfccs })
    }
}
