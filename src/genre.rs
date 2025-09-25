use bincode::{Decode, Encode};

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
pub enum Genre {
    Rock,
    HipHop,
    Electronic,
    Classical,
}

impl Genre {
    pub fn from_string(string: &str) -> Genre {
        match string {
            "rock" => Genre::Rock,
            "hiphop" => Genre::HipHop,
            "electronic" => Genre::Electronic,
            "classical" => Genre::Classical,
            _ => panic!("Unexpected genre."),
        }
    }

    pub fn index(&self) -> u32 {
        match self {
            Genre::Rock => 0,
            Genre::HipHop => 1,
            Genre::Electronic => 2,
            Genre::Classical => 3,
        }
    }
}

impl Genre {
    pub const GENRES_N: usize = 4;
}
