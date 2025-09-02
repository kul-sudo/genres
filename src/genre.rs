use bincode::{Decode, Encode};

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq, Eq, Hash)]
pub enum Genre {
    Punk,
    RockNRoll,
    Pop,
    Electronic,
}

impl Genre {
    pub const GENRES_N: usize = 4;
}

impl From<Genre> for i64 {
    fn from(genre: Genre) -> i64 {
        match genre {
            Genre::Punk => 0,
            Genre::RockNRoll => 1,
            Genre::Pop => 2,
            Genre::Electronic => 3,
        }
    }
}

impl From<String> for Genre {
    fn from(string: String) -> Genre {
        match string.as_str() {
            "punk" => Genre::Punk,
            "rocknroll" => Genre::RockNRoll,
            "pop" => Genre::Pop,
            "electronic" => Genre::Electronic,
            _ => panic!(),
        }
    }
}
