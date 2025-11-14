use bincode::{Decode, Encode};

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
pub enum Genre {
    Classical,
    Electronic,
    HipHop,
    Rock,
}

impl Genre {
    pub fn from_string(string: &str) -> Option<Genre> {
        match string.to_lowercase().as_str() {
            "classical" => Some(Genre::Classical),
            "electronic" => Some(Genre::Electronic),
            "hip-hop" => Some(Genre::HipHop),
            "rock" => Some(Genre::Rock),
            _ => {
                println!("Unexpected genre: {}", string);
                None
            }
        }
    }
}

impl Genre {
    pub const GENRES_N: usize = 4;
}
