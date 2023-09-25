pub trait GraphType {
    fn is_directed() -> bool;
}

pub struct Directed;

impl GraphType for Directed {
    fn is_directed() -> bool {
        true
    }
}

pub struct UnDirected;

impl GraphType for UnDirected {
    fn is_directed() -> bool {
        false
    }
}

#[derive(PartialEq, Eq, Clone)]
pub enum Direction {
    Outcoming,
    Incoming,
}

impl Direction {
    pub fn oppsite(&self) -> Self {
        match self {
            Self::Outcoming => Self::Incoming,
            Self::Incoming => Self::Outcoming,
        }
    }
}

impl From<Direction> for usize {
    fn from(value: Direction) -> Self {
        if let Direction::Outcoming = value {
            0
        } else {
            1
        }
    }
}

impl From<usize> for Direction {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Outcoming,
            1 => Self::Incoming,
            _ => panic!("only 0 and 1 are valid num"),
        }
    }
}
