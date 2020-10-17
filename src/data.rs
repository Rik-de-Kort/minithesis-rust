/// Represents some range of values that might be used in a test, that can be requested from a
/// TestCase.
// use crate::*;
use crate::test_harness::*;
use std::clone::Clone;
use std::convert::TryInto;
use std::marker::{PhantomData, Sized};

pub trait Possibility<T>: Sized {
    fn produce(&self, tc: &mut TestCase) -> Result<T, MTErr>;

    fn map<U, F: Fn(T) -> U>(self, f: F) -> Map<T, U, F, Self> {
        Map {
            source: self,
            map: f,
            phantom_t: PhantomData,
            phantom_u: PhantomData,
        }
    }

    fn bind<U, F: Fn(T) -> Q, Q: Possibility<U>>(self, f: F) -> Bind<T, U, F, Self, Q> {
        Bind {
            source: self,
            map: f,
            phantom_t: PhantomData,
            phantom_u: PhantomData,
            phantom_q: PhantomData,
        }
    }

    fn satisfying<F: Fn(&T) -> bool>(self, f: F) -> Satisfying<T, F, Self> {
        Satisfying {
            source: self,
            predicate: f,
            phantom_t: PhantomData,
        }
    }
}

pub struct Map<T, U, F: Fn(T) -> U, P: Possibility<T>> {
    source: P,
    map: F,
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
}
impl<T, U, F: Fn(T) -> U, P: Possibility<T>> Possibility<U> for Map<T, U, F, P> {
    fn produce(&self, tc: &mut TestCase) -> Result<U, MTErr> {
        Ok((self.map)(tc.any(&self.source)?))
    }
}

pub struct Bind<T, U, F: Fn(T) -> Q, P: Possibility<T>, Q: Possibility<U>> {
    source: P,
    map: F,
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
    phantom_q: PhantomData<Q>,
}
impl<T, U, F: Fn(T) -> Q, P: Possibility<T>, Q: Possibility<U>> Bind<T, U, F, P, Q> {
    fn produce(&self, tc: &mut TestCase) -> Result<U, MTErr> {
        let inner = tc.any(&self.source)?;
        tc.any(&(self.map)(inner))
    }
}

pub struct Satisfying<T, F: Fn(&T) -> bool, P: Possibility<T>> {
    source: P,
    predicate: F,
    phantom_t: PhantomData<T>,
}
impl<T, F: Fn(&T) -> bool, P: Possibility<T>> Possibility<T> for Satisfying<T, F, P> {
    fn produce(&self, tc: &mut TestCase) -> Result<T, MTErr> {
        for _ in 0..3 {
            let candidate = tc.any(&self.source)?;
            if (self.predicate)(&candidate) {
                return Ok(candidate);
            }
        }
        Err(tc.reject())
    }
}

pub struct Integers {
    minimum: i64,
    range: u64,
}
impl Integers {
    pub fn new(minimum: i64, maximum: i64) -> Self {
        Integers {
            minimum,
            range: (maximum - minimum).try_into().unwrap(),
        }
    }
}

impl Possibility<i64> for Integers {
    fn produce(&self, tc: &mut TestCase) -> Result<i64, MTErr> {
        let offset: i64 = tc.choice(self.range)?.try_into().unwrap();
        Ok(self.minimum + offset)
    }
}

pub struct Vectors<U, T: Possibility<U>> {
    elements: T,
    min_size: usize,
    max_size: usize,
    phantom: PhantomData<U>, // Required for type check
}
impl<U, T: Possibility<U>> Vectors<U, T> {
    pub fn new(elements: T, min_size: usize, max_size: usize) -> Self {
        Vectors {
            elements,
            min_size,
            max_size,
            phantom: PhantomData,
        }
    }
}

impl<U, T: Possibility<U>> Possibility<Vec<U>> for Vectors<U, T> {
    fn produce(&self, tc: &mut TestCase) -> Result<Vec<U>, MTErr> {
        let mut result = vec![];
        loop {
            if result.len() < self.min_size {
                tc.det_choice(1)?;
            } else if result.len() + 1 >= self.max_size {
                tc.det_choice(0)?;
                break;
            } else if tc.weighted(0.9)? == 0 {
                break;
            }
            result.push(tc.any(&self.elements)?);
        }
        Ok(result)
    }
}

pub struct Just<T: Clone> {
    value: T,
}

impl<T: Clone> Possibility<T> for Just<T> {
    fn produce(&self, _: &mut TestCase) -> Result<T, MTErr> {
        Ok(self.value.clone())
    }
}

pub struct Nothing {}
impl<T> Possibility<T> for Nothing {
    fn produce(&self, tc: &mut TestCase) -> Result<T, MTErr> {
        Err(tc.reject())
    }
}

pub struct MixOf<T, P: Possibility<T>> {
    first: P,
    second: P,
    phantom_t: PhantomData<T>,
}

impl<T, P: Possibility<T>> MixOf<T, P> {
    pub fn new(first: P, second: P) -> Self {
        MixOf {
            first,
            second,
            phantom_t: PhantomData,
        }
    }
}

impl<T, P: Possibility<T>> Possibility<T> for MixOf<T, P> {
    fn produce(&self, tc: &mut TestCase) -> Result<T, MTErr> {
        if tc.choice(1)? == 0 {
            tc.any(&self.first)
        } else {
            tc.any(&self.second)
        }
    }
}

pub fn vectors<U, T: Possibility<U>>(
    elements: T,
    min_size: usize,
    max_size: usize,
) -> Vectors<U, T> {
    Vectors::new(elements, min_size, max_size)
}

pub fn integers(minimum: i64, maximum: i64) -> Integers {
    Integers::new(minimum, maximum)
}

pub fn just<T: Clone>(value: T) -> Just<T> {
    Just { value }
}
pub fn nothing() -> Nothing {
    Nothing {}
}
pub fn mix_of<T, P: Possibility<T>>(first: P, second: P) -> MixOf<T, P> {
    MixOf::new(first, second)
}

