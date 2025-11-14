use crate::{audio::*, consts::*, genre::*};
use std::fs::read_dir;

pub fn files_init() -> Vec<Vec<Crop>> {
    let mut files = vec![];

    let total = read_dir(INPUT_DIR)
        .unwrap()
        .map(|entry| read_dir(entry.unwrap().path()).unwrap().count())
        .sum::<usize>();

    let mut n = 0;

    for entry in read_dir(INPUT_DIR).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap();
        let genre = Genre::from_string(name.to_str().unwrap());

        if let Some(g) = genre {
            for audio_entry in read_dir(&path).unwrap() {
                let mut crops = Vec::with_capacity(N_CROPS.get());
                let audio_entry = audio_entry.unwrap();
                let audio_path = audio_entry.path();

                for _ in 0..N_CROPS.get() {
                    if let Some(variations) = Crop::prepare(&audio_path) {
                        for variation in variations {
                            crops.push(Crop::new(variation, g))
                        }
                    }
                }

                files.push(crops);
                n += 1;
                println!("{:.2}% done.", n as f32 / total as f32 * 100.0)
            }
        }
    }

    files
}
