use crate::{audio::*, consts::*, genre::*};
use std::fs::read_dir;

pub fn files_init() -> Vec<Vec<Crop>> {
    let mut files = vec![];

    for entry in read_dir(INPUT_DIR).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap();
        let genre = Genre::from_string(name.to_str().unwrap());

        for audio_entry in read_dir(&path).unwrap() {
            let mut crops = Vec::with_capacity(CROPS_N.get());
            let audio_entry = audio_entry.unwrap();
            let audio_path = audio_entry.path();

            for _ in 0..CROPS_N.get() {
                if let Some(data) = Crop::prepare(&audio_path) {
                    crops.push(Crop::new(data, genre))
                }
            }

            files.push(crops);
        }
    }

    files
}
