use bincode::{Decode, Encode};

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
pub enum Genre {
    Rock,
    Electronic,
    Classical,
    Jazz,
}

impl Genre {
    pub fn from_string(string: &str) -> Genre {
        match string {
            "rock" => Genre::Rock,
            "electronic" => Genre::Electronic,
            "classical" => Genre::Classical,
            "jazz" => Genre::Jazz,
            _ => panic!("Unexpected genre."),
        }
    }

    pub fn index(&self) -> u32 {
        match self {
            Genre::Rock => 0,
            Genre::Electronic => 1,
            Genre::Classical => 2,
            Genre::Jazz => 3,
        }
    }
}

impl Genre {
    pub const GENRES_N: usize = 4;
}
