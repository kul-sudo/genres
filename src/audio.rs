use crate::{consts::*, genre::*};
use rand::{Rng, rng};
use std::{fs::File, path::PathBuf};

use bincode::{Decode, Encode};
use mel_spec::{mel::MelSpectrogram, stft::Spectrogram};
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

    pub fn prepare(source: &PathBuf) -> Option<Vec<f32>> {
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

        let mut stft = Spectrogram::new(FFT_SIZE, HOP_SIZE);
        let mut mel_transform = MelSpectrogram::new(FFT_SIZE, SAMPLE_RATE as f64, N_MELS);

        let mut mel_spectrogram = Vec::new();

        for chunk in slice.chunks(HOP_SIZE) {
            if let Some(fft_frame) = stft.add(chunk) {
                let mel_frame = mel_transform.add(&fft_frame);

                mel_spectrogram.extend(mel_frame.iter().map(|&x| x as f32));
            }
        }

        assert!(!mel_spectrogram.iter().any(|x: &f32| x.is_nan()));
        assert_eq!(mel_spectrogram.len(), N_FRAMES * N_MELS);

        Some(mel_spectrogram)
    }
}
