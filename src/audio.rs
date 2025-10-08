use crate::{consts::*, genre::*};
use rand::{Rng, rng};
use std::{fs::File, path::PathBuf};

use aubio::{MFCC, PVoc, carr, farr};
use bincode::{Decode, Encode};
use hound::{SampleFormat, WavSpec, WavWriter};
use pitch_shift::PitchShifter;
use reverb::Reverb;
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

        assert!(samples.len() >= CROP_PAD);

        let crop_start = rng().random_range(0..samples.len() - CROP_PAD);
        let slice = &mut samples[crop_start..crop_start + CROP_PAD];

        let mut variations = Vec::with_capacity(N_VARIATIONS + 1);

        for _ in 0..N_VARIATIONS {
            let mut shifter = PitchShifter::new(50, SAMPLE_RATE);
            let semitones = rng().random_range(SEMITONE_VARIANCE);
            let mut shifted = vec![0.0; slice.len()];
            shifter.shift_pitch(16, semitones, slice, &mut shifted);

            if rng().random_range(0.0..=1.0) < DISTORTION_CHANCE {
                let mut reverb = Reverb::new();
                reverb.decay(DECAY).damping(DAMPING).bandwidth(BANDWIDTH);

                let dry_level = rng().random_range(DRY_RANGE);
                let wet_level = rng().random_range(WET_RANGE);

                for sample in shifted.iter_mut() {
                    let wet_sample = reverb.calc_sample(*sample, wet_level);
                    *sample = *sample * dry_level + wet_sample;
                }

                for sample in shifted.iter_mut() {
                    *sample += rng().random_range(-NOISE_LEVEL..NOISE_LEVEL);
                }
            }

            variations.push(process(&mut shifted));
        }

        variations.push(process(slice));

        Some(variations)
    }
}

fn save_wav(path: &str, samples: &[f32], sample_rate: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

fn process(input: &mut [f32]) -> Vec<f32> {
    let rms = (input.iter().map(|x| x.powi(2)).sum::<f32>() / input.len() as f32 + 1e-8).sqrt();
    for sample in input.iter_mut() {
        *sample /= rms;
    }

    assert!(input.len() >= FRAME_SIZE);

    let mut pvoc = PVoc::new(FRAME_SIZE, HOP_LENGTH).unwrap();
    let mut mfcc = MFCC::new(FRAME_SIZE, N_FILTERS, N_COEFFS, SAMPLE_RATE as u32)
        .unwrap()
        .with_mel_coeffs_slaney();

    let mut fftgrain = carr!(FRAME_SIZE);
    let mut mfcc_out = farr!(N_COEFFS);
    let mut all_mfccs = Vec::with_capacity(MAX_SAMPLES_N);

    for chunk in input.chunks_exact(HOP_LENGTH) {
        pvoc.do_(chunk, &mut fftgrain).unwrap();
        mfcc.do_(fftgrain, &mut mfcc_out).unwrap();
        all_mfccs.extend_from_slice(mfcc_out.as_slice());
    }

    assert_eq!(all_mfccs.len(), MAX_SAMPLES_N);
    all_mfccs
}
