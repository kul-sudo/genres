use crate::{CONFIG, Mode, consts::*, genre::*};
use rand::{Rng, rng};
use std::{fs::File, path::PathBuf};

use bincode::{Decode, Encode};
use biquad::{Biquad, Coefficients, DirectForm2Transposed, Q_BUTTERWORTH_F32, ToHertz, Type};
use hound::{SampleFormat, WavSpec, WavWriter};
use mel_spec::{mel::MelSpectrogram, stft::Spectrogram};
use ndarray::Array;
use signalsmith_stretch::Stretch;
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

#[derive(Clone, Debug, Encode, Decode)]
pub struct Crop {
    data: Vec<f32>,
    genre: Genre,
}

impl Crop {
    pub fn new(data: Vec<f32>, genre: Genre) -> Crop {
        Crop { data, genre }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn genre(&self) -> &Genre {
        &self.genre
    }

    pub fn normalize(&mut self, mean: f32, std: f32) {
        normalize(&mut self.data, mean, std);
    }

    pub fn prepare(source: &PathBuf) -> Option<Vec<Vec<f32>>> {
        let src = File::open(source).unwrap();
        let mss = MediaSourceStream::new(Box::new(src), Default::default());

        let mut hint = Hint::new();
        hint.with_extension("ogg");

        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)
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
                samples.push(mono);
            }
        }

        if samples.len() < CROP_PAD {
            println!("{} skipped.", source.file_name().unwrap().to_str().unwrap());
            return None;
        };

        let crop_start = rng().random_range(0..samples.len() - CROP_PAD);
        let slice: [f32; CROP_PAD] = samples[crop_start..crop_start + CROP_PAD]
            .try_into()
            .unwrap();

        let mut variations = vec![];

        let mut local = slice;
        let mut applied = false;

        match CONFIG.mode {
            Mode::Train => {
                if rng().random_range(0.0..=1.0) < SIGNALSMITH_CHANCE_TRAINING {
                    signalsmith(&mut local);
                    applied = true;
                }

                if rng().random_range(0.0..=1.0) < LOFI_CHANCE_TRAINING {
                    lofi(&mut local);
                    applied = true;
                }
            }
            Mode::Test => {
                if rng().random_range(0.0..=1.0) < SIGNALSMITH_CHANCE_TESTING {
                    signalsmith(&mut local);
                    applied = true;
                }

                if rng().random_range(0.0..=1.0) < LOFI_CHANCE_TESTING {
                    lofi(&mut local);
                    applied = true;
                }
            }
        }

        if applied {
            // save_wav(
            //     source.file_name().unwrap().to_str().unwrap(),
            //     &local,
            //     SAMPLE_RATE as u32,
            // );
            variations.push(process(&local));
        }

        variations.push(process(&slice));

        Some(variations)
    }
}

fn signalsmith(input: &mut [f32; CROP_PAD]) {
    let mut stretch = Stretch::new(1, 1024, 256);
    let shift = rng().random_range(PITCH_SHIFT_RANGE);
    stretch.set_transpose_factor_semitones(shift, None);
    stretch.set_formant_factor_semitones(-shift, true);

    let output_len = (input.len() as f32 * rng().random_range(SPEED_STRETCH_RANGE)) as usize;
    let mut output = vec![0.0; output_len];

    stretch.exact(&input, &mut output);
    output.resize(CROP_PAD, 0.0);
    input.copy_from_slice(&output);
}

fn lofi(input: &mut [f32; CROP_PAD]) {
    let mut hpf = DirectForm2Transposed::<f32>::new(
        Coefficients::<f32>::from_params(
            Type::HighPass,
            SAMPLE_RATE.hz(),
            rng().random_range(HIGH_PASS_RANGE).hz(),
            Q_BUTTERWORTH_F32,
        )
        .unwrap(),
    );

    let mut lpf = DirectForm2Transposed::<f32>::new(
        Coefficients::<f32>::from_params(
            Type::LowPass,
            SAMPLE_RATE.hz(),
            rng().random_range(LOW_PASS_RANGE).hz(),
            Q_BUTTERWORTH_F32,
        )
        .unwrap(),
    );

    for sample in input.iter_mut() {
        *sample = hpf.run(*sample);
        *sample = lpf.run(*sample);
        *sample += rng().random_range(NOISE_RANGE);
    }
}

fn save_wav(path: &str, samples: &[f32], sample_rate: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for &sample in samples {
        writer
            .write_sample((sample * i16::MAX as f32) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

fn process(input: &[f32; CROP_PAD]) -> Vec<f32> {
    let mut stft = Spectrogram::new(FRAME_SIZE, HOP_LENGTH);
    let mut mel = MelSpectrogram::new(FRAME_SIZE, SAMPLE_RATE as f64, N_FILTERS);

    let expected_len = N_FILTERS * ((CROP_PAD - FRAME_SIZE) / HOP_LENGTH + 1);

    let mut all_mels = Vec::with_capacity(expected_len);
    let rms = (input.iter().map(|x| x.powi(2)).sum::<f32>() / input.len() as f32 + 1e-8).sqrt();

    for chunk in input
        .iter()
        .map(|&x| x / rms)
        .collect::<Vec<_>>()
        .chunks_exact(HOP_LENGTH)
    {
        if let Some(fft) = stft.add(chunk) {
            let mel_frame = mel.add(&fft);
            all_mels.extend_from_slice(mel_frame.mapv(|x| x as f32).as_slice().unwrap());
        }
    }

    assert_eq!(all_mels.len(), expected_len);

    all_mels
}

pub fn compute_global_stats(items: &[Vec<Crop>]) -> (f32, f32) {
    let mut all_values = Vec::new();
    for crops in items {
        for crop in crops {
            all_values.extend_from_slice(crop.data());
        }
    }

    let tensor = Array::from_shape_vec(all_values.len(), all_values).unwrap();
    let mean = tensor.mean().unwrap();
    let std = tensor.std(0.0);

    assert!(!mean.is_nan() && !std.is_nan());

    (mean, std)
}

pub fn normalize(input: &mut [f32], mean: f32, std: f32) {
    for val in input {
        *val = (*val - mean) / (std + 1e-8);
    }
}
