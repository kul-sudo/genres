use crate::{consts::*, genre::*};
use rand::{Rng, rng};
use std::{fs::File, path::PathBuf};

use bincode::{Decode, Encode};
use mfcc::Transform;
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
    data: Vec<f64>,
    genre: Genre,
}

impl Crop {
    pub fn new(data: Vec<f64>, genre: Genre) -> Crop {
        Crop { data, genre }
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn genre(&self) -> &Genre {
        &self.genre
    }

    pub fn prepare(source: &PathBuf) -> Option<Vec<f64>> {
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

        let rms = (slice.iter().map(|x| x.powi(2)).sum::<f32>() / slice.len() as f32 + 1e-8).sqrt();
        for sample in slice.iter_mut() {
            *sample /= rms;
        }

        let slice = slice
            .iter()
            .map(|sample| (sample * i16::MAX as f32) as i16)
            .collect::<Vec<_>>();

        let mut state = Transform::new(SAMPLE_RATE, FRAME_SIZE).nfilters(N_COEFFS, N_FILTERS);

        let mut all_mfccs = Vec::with_capacity(MAX_SAMPLES_N);
        for chunk in slice.chunks_exact(FRAME_SIZE) {
            let mut output = vec![0.0; N_COEFFS * 3];
            state.transform(chunk, &mut output);
            all_mfccs.extend_from_slice(&output);
        }

        assert_eq!(all_mfccs.len(), MAX_SAMPLES_N);

        Some(all_mfccs)
    }
}
