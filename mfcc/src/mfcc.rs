use crate::freqs::{ForwardRealFourier, InverseCosineTransform};
use crate::ringbuffer::Ringbuffer;
#[cfg(any(feature = "fftw", feature = "fftextern"))]
use fftw::array::AlignedVec;
#[cfg(feature = "fftrust")]
use num_complex::Complex;
use num_complex::Complex64;
use std::collections::VecDeque;

#[cfg(feature = "fftrust")]
type AlignedVec<T> = Vec<T>;

pub struct Transform {
    idct: InverseCosineTransform,
    rfft: ForwardRealFourier,
    rb: Ringbuffer,

    windowed_samples: AlignedVec<f64>,
    samples_freq_domain: AlignedVec<Complex64>,
    filters: AlignedVec<f64>,
    mean_coeffs: Vec<f64>,
    prev_coeffs: VecDeque<Vec<f64>>,

    maxmel: f64,

    sample_rate: usize,
    maxfilter: usize,
    nfilters: usize,
    buffer_size: usize,
}

impl Transform {
    pub fn new(sample_rate: usize, buffer_size: usize) -> Transform {
        //let size = (2 * buffer_size).next_power_of_two();
        let size = 2 * buffer_size;
        let nfilters = 40;
        let maxfilter = 16;
        #[cfg(feature = "fftrust")]
        let windowed_samples = vec![0.0; size];
        #[cfg(feature = "fftrust")]
        let samples_freq_domain = vec![Complex::i(); size / 2 + 1];
        #[cfg(feature = "fftrust")]
        let filters = vec![0.0; nfilters];

        #[cfg(any(feature = "fftw", feature = "fftextern"))]
        let windowed_samples = AlignedVec::new(size);
        #[cfg(any(feature = "fftw", feature = "fftextern"))]
        let samples_freq_domain = AlignedVec::new(size / 2 + 1);
        #[cfg(any(feature = "fftw", feature = "fftextern"))]
        let filters = AlignedVec::new(nfilters);

        Transform {
            idct: InverseCosineTransform::new(nfilters),
            rfft: ForwardRealFourier::new(size),
            rb: Ringbuffer::new(2 * buffer_size),

            windowed_samples,
            samples_freq_domain,
            filters,
            mean_coeffs: vec![0.0; maxfilter * 3],
            prev_coeffs: VecDeque::new(),

            maxmel: 2595.0 * (1.0 + sample_rate as f64 / 2.0 / 700.0).log10(),

            sample_rate,
            maxfilter,
            nfilters,
            buffer_size,
        }
    }

    pub fn nfilters(mut self, maxfilter: usize, nfilters: usize) -> Transform {
        self.maxfilter = maxfilter;
        self.nfilters = nfilters;

        #[cfg(feature = "fftrust")]
        let filters = vec![0.0; nfilters];
        #[cfg(any(feature = "fftw", feature = "fftextern"))]
        let filters = AlignedVec::new(nfilters);

        self.filters = filters;

        self.mean_coeffs = vec![0.0; maxfilter * 3];

        self
    }

    pub fn transform(&mut self, input: &[i16], output: &mut [f64]) {
        assert_eq!(input.len(), self.buffer_size);
        assert_eq!(output.len(), self.maxfilter * 3);

        self.rb.append_back(&input);
        self.rb.apply_hamming(&mut self.windowed_samples);

        self.rfft
            .transform(&mut self.windowed_samples, &mut self.samples_freq_domain);

        for x in self.filters.iter_mut() {
            *x = 0.0;
        }

        let filter_length = (self.maxmel / self.nfilters as f64) * 2.0;

        for (idx, val) in self.samples_freq_domain.iter().skip(1).enumerate() {
            let mel = 2595.0
                * (1.0
                    + (self.sample_rate as f64 / 2.0 * (1.0 + idx as f64)
                        / (self.samples_freq_domain.len() as f64))
                        / 700.0)
                    .log10();
            let mut idx = ((mel / self.maxmel) * self.nfilters as f64).floor() as usize;
            let val = (val.re / self.windowed_samples.len() as f64).powf(2.0)
                + (val.im / self.windowed_samples.len() as f64).powf(2.0);

            if idx == self.nfilters {
                idx -= 1;
            }

            // push to previous filterbank (ignore special case in first bank)
            if idx > 0 {
                // calculate position from beginning of the filter
                let mel_diff = mel - (idx - 1) as f64 * filter_length / 2.0;
                // normalize to range [0.0, 1.0]
                let mel_diff = mel_diff / filter_length;

                //if mel_diff < 0.5 {
                //    filters[idx-1] += mel_diff * val;
                //} else {
                self.filters[idx - 1] += (1.0 - mel_diff) * val;
                //}
            }

            // calculate position from beginning of the filter
            let mel_diff = mel - idx as f64 * filter_length / 2.0;
            // normalize to range [0.0, 1.0]
            let mel_diff = mel_diff / filter_length;

            //if mel_diff < 0.5 {
            self.filters[idx] += mel_diff * val;
            //} else {
            //    filters[idx] = (1.0 - mel_diff) * val;
            //}
        }

        for filter in self.filters.iter_mut() {
            if *filter < 1e-20 {
                *filter = -46.05;
            } else {
                *filter = (*filter).ln();
            }
        }

        self.idct.transform(&mut self.filters, output);

        for i in self.maxfilter..output.len() {
            output[i] = 0.0;
        }

        if let Some(back) = self.prev_coeffs.back() {
            for i in 0..self.maxfilter {
                output[self.maxfilter + i] = output[i] - back[i];
                output[self.maxfilter * 2 + i] =
                    output[self.maxfilter + i] - back[self.maxfilter + i];
            }
        }
    }
}
