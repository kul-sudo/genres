use crate::audio::*;
use crate::consts::*;
use crate::genre::*;
use std::{collections::HashMap, fs::read_dir, path::PathBuf, sync::LazyLock};

type Entries = Vec<(PathBuf, Option<MfccData>)>;

pub static FILES: LazyLock<HashMap<Genre, Entries>> = LazyLock::new(|| {
    let mut files = HashMap::with_capacity(Genre::GENRES_N);

    for entry in read_dir(GENRES_DIR).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = path.file_name().unwrap();
        let genre = Genre::from(name.to_string_lossy().to_string());

        for audio in read_dir(&path).unwrap() {
            let audio = audio.unwrap();
            let audio_path = audio.path();

            let data = MfccData::new(MfccSource::Path(path.to_path_buf()));

            files
                .entry(genre)
                .and_modify(|x: &mut Vec<(PathBuf, Option<MfccData>)>| {
                    x.push((audio_path.clone(), data.clone()))
                })
                .or_insert(vec![(audio_path.clone(), data)]);
        }
    }

    files
});
