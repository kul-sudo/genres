use crate::{audio::*, consts::*, genre::*};
use std::{collections::HashMap, fs::read_dir, path::PathBuf, sync::LazyLock};

#[derive(Clone)]
pub struct Audio {
    data: Option<MfccData>,
}

impl Audio {
    pub fn data(&self) -> &Option<MfccData> {
        &self.data
    }
}

pub static FILES: LazyLock<HashMap<Genre, Vec<Audio>>> = LazyLock::new(|| {
    let mut files = HashMap::with_capacity(Genre::GENRES_N);

    for entry in read_dir(GENRES_DIR).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap();
        let genre = Genre::from(name.to_string_lossy().to_string());

        for audio_entry in read_dir(&path).unwrap() {
            let audio_entry = audio_entry.unwrap();
            let audio_path = audio_entry.path();

            for _ in 0..CROPS_N {
                let data = MfccData::new(MfccSource::Path(audio_path.clone().into_boxed_path()));

                let audio = Audio { data };

                files
                    .entry(genre)
                    .and_modify(|x: &mut Vec<Audio>| x.push(audio.clone()))
                    .or_insert(vec![audio]);
            }
        }
    }

    files
});
