use bincode::{Decode, Encode};

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
pub enum Genre {
    Rock,
    Electronic,
    Rap,
    Classical,
}

impl Genre {
    pub fn from_string(string: &str) -> Genre {
        match string {
            "rock" => Genre::Rock,
            "electronic" => Genre::Electronic,
            "rap" => Genre::Rap,
            "classical" => Genre::Classical,
            _ => panic!("Unexpected genre."),
        }
    }

    pub fn index(&self) -> u32 {
        match self {
            Genre::Rock => 0,
            Genre::Electronic => 1,
            Genre::Rap => 2,
            Genre::Classical => 3,
        }
    }
}

impl Genre {
    pub const GENRES_N: usize = 4;
}
