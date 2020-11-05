use rand::prelude::*;
mod database;

#[derive(Debug)]
pub enum Error {
    Overrun,
    Invalid,
}

#[derive(Debug, PartialEq)]
enum Status {
    Invalid,
    Valid,
    Interesting,
}

#[derive(Debug)]
pub struct TestCase {
    prefix: Vec<u64>,
    random: ThreadRng,
    max_size: usize,
    choices: Vec<u64>,
    status: Option<Status>,
    depth: u64,
}

use crate::data::Possibility;
impl TestCase {
    fn new(prefix: Vec<u64>, random: ThreadRng, max_size: usize) -> TestCase {
        TestCase {
            prefix,
            random,
            max_size,
            choices: vec![],
            status: None,
            depth: 0,
        }
    }

    fn for_choices(prefix: Vec<u64>) -> TestCase {
        TestCase {
            max_size: prefix.len(),
            prefix,
            random: thread_rng(),
            choices: vec![],
            status: None,
            depth: 0,
        }
    }

    /// Insert a definite choice in the choice sequence
    /// N.B. All integrity checks happen here!
    fn forced_choice(&mut self, n: u64) -> Result<u64, Error> {
        if self.choices.len() >= self.max_size {
            Err(Error::Overrun)
        } else {
            self.choices.push(n);
            Ok(n)
        }
    }

    /// Return 1 with probability p, 0 otherwise.
    fn weighted(&mut self, p: f64) -> Result<u64, Error> {
        if self.choices.len() < self.prefix.len() {
            if self.prefix[self.choices.len()] > 1 {
                Err(Error::Invalid)
            } else {
                self.forced_choice(self.prefix[self.choices.len()])
            }
        } else if self.random.gen_bool(p) {
            self.forced_choice(1)
        } else {
            self.forced_choice(0)
        }
    }

    /// Mark this test case as invalid
    fn reject(&mut self) -> Error {
        Error::Invalid
    }

    /// If this precondition is not met, abort the test and mark this test case as invalid
    fn assume(&mut self, precondition: bool) -> Option<Error> {
        if !precondition {
            Some(self.reject())
        } else {
            None
        }
    }

    /// Return an integer in the range [0, n]
    fn choice(&mut self, n: u64) -> Result<u64, Error> {
        if self.choices.len() < self.prefix.len() {
            self.forced_choice(self.prefix[self.choices.len()])
        } else {
            let result = self.random.gen_range(0, n + 1);
            self.forced_choice(result)
        }
    }
}

mod data {
    /// Represents some range of values that might be used in a test, that can be requested from a
    /// TestCase.
    use crate::*;
    use std::clone::Clone;
    use std::convert::TryInto;
    use std::marker::{PhantomData, Sized};

    pub trait Possibility<T>: Sized {
        fn produce(&self, tc: &mut TestCase) -> Result<T, Error>;

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
        fn produce(&self, tc: &mut TestCase) -> Result<U, Error> {
            Ok((self.map)(self.source.produce(tc)?))
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
        fn produce(&self, tc: &mut TestCase) -> Result<U, Error> {
            let inner = self.source.produce(tc)?;
            (self.map)(inner).produce(tc)
        }
    }

    pub struct Satisfying<T, F: Fn(&T) -> bool, P: Possibility<T>> {
        source: P,
        predicate: F,
        phantom_t: PhantomData<T>,
    }
    impl<T, F: Fn(&T) -> bool, P: Possibility<T>> Possibility<T> for Satisfying<T, F, P> {
        fn produce(&self, tc: &mut TestCase) -> Result<T, Error> {
            for _ in 0..3 {
                let candidate = self.source.produce(tc)?;
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
        fn produce(&self, tc: &mut TestCase) -> Result<i64, Error> {
            let offset: i64 = tc.choice(self.range)?.try_into().unwrap();
            println!("got integer choice {}", offset);
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
        fn produce(&self, tc: &mut TestCase) -> Result<Vec<U>, Error> {
            let mut result = vec![];
            loop {
                if result.len() < self.min_size {
                    tc.forced_choice(1)?;
                } else if result.len() + 1 >= self.max_size {
                    tc.forced_choice(0)?;
                    break;
                } else if tc.weighted(0.9)? == 0 {
                    break;
                }
                result.push(self.elements.produce(tc)?);
            }
            Ok(result)
        }
    }

    pub struct Just<T: Clone> {
        value: T,
    }

    impl<T: Clone> Possibility<T> for Just<T> {
        fn produce(&self, _: &mut TestCase) -> Result<T, Error> {
            Ok(self.value.clone())
        }
    }

    pub struct Nothing {}
    impl<T> Possibility<T> for Nothing {
        fn produce(&self, tc: &mut TestCase) -> Result<T, Error> {
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
        fn produce(&self, tc: &mut TestCase) -> Result<T, Error> {
            if tc.choice(1)? == 0 {
                self.first.produce(tc)
            } else {
                self.second.produce(tc)
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
}


struct TestState {
    random: ThreadRng,
    max_examples: usize,
    is_interesting: Box<dyn Fn(&mut TestCase) -> Status>,
    valid_test_cases: usize,
    calls: usize,
    result: Option<Vec<u64>>,
    best_scoring: Option<Vec<u64>>, // Todo: check type
    test_is_trivial: bool,
}

const BUFFER_SIZE: usize = 8 * 1024;

impl TestState {
    pub fn new(
        random: ThreadRng,
        test_function: Box<dyn Fn(&mut TestCase) -> Status>,
        max_examples: usize,
    ) -> TestState {
        TestState {
            random,
            is_interesting: test_function,
            max_examples,
            valid_test_cases: 0,
            calls: 0,
            result: None,
            best_scoring: None,
            test_is_trivial: false,
        }
    }

    fn test_function(&mut self, mut test_case: &mut TestCase) -> bool {
        self.calls += 1;

        match (self.is_interesting)(&mut test_case) {
            Status::Valid => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;
                false
            },
            Status::Invalid => {false},
            Status::Interesting => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;

                if self.result == None || *self.result.as_ref().unwrap() > test_case.choices {
                    self.result = Some(test_case.choices.clone());
                    true
                } else {
                    false
                }
            }
        }
    }

    fn result_as<T>(&self, p: impl Possibility<T>) -> Option<T> {
        if let Some(choices) = &self.result {
            Some(p.produce(&mut TestCase::for_choices(choices.to_vec())).unwrap())
        } else {
            None
        }
    }

    fn run(&mut self) {
        self.generate();
        self.shrink();
    }

    fn should_keep_generating(&self) -> bool {
        (!self.test_is_trivial)
            & (self.result == None)
            & (self.valid_test_cases < self.max_examples)
            & (self.calls < self.max_examples * 10)
    }

    fn generate(&mut self) {
        while self.should_keep_generating()
            & ((self.best_scoring == None) || self.valid_test_cases <= self.max_examples / 2)
        {
            self.test_function(&mut TestCase::new(vec![], self.random, BUFFER_SIZE));
        }
    }

    fn consider(&mut self, choices: &[u64]) -> bool {
        if Some(choices.to_vec()) == self.result {
            true
        } else {
            self.test_function(&mut TestCase::for_choices(choices.to_vec()))
        }
    }

    fn shrink_remove(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if k > attempt.len() {
            return None;
        }

        // Generate all valid removals (don't worry, it's lazy!)
        let valid = (0..=attempt.len()-k).map(|j| (j, j+k)).rev();
        for (x, y) in valid {
            let head = &attempt[..x];
            let tail = if y < attempt.len() { &attempt[y..] } else { &[] };
            let mut new = [head, tail].concat();

            if self.consider(&new) {
                return Some(new);
            } else if x > 0 && new[x - 1] > 0 {
                // Short-circuit prevents overflow
                new[x - 1] -= 1;
                if self.consider(&new) {
                    return Some(new);
                };
            }
        }
        None
    }

    fn shrink_zeroes(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if k > attempt.len() {
            return None;
        }
        let valid = (k..attempt.len() - 1).map(|j| (j - k, j)).rev();
        for (x, y) in valid {
            if attempt[x..y].iter().all(|i| *i == 0) {
                continue;
            }
            let new = [&attempt[..x], &vec![0; y - x], &attempt[y..]].concat();
            if self.consider(&new) {
                return Some(new);
            }
        }
        None
    }

    fn shrink_reduce(&mut self, attempt: &[u64]) -> Option<Vec<u64>> {
        let mut new = attempt.to_owned();
        for i in (0..attempt.len()).rev() {
            let mut low = 0;
            let mut high = new[i];
            while low + 1 < high {
                let mid = low + (high - low) / 2;
                new[i] = mid;
                if self.consider(&new) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            new[i] = high;
        }
        if new == attempt {
            None
        } else {
            Some(new)
        }
    }

    fn shrink_sort(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if k > attempt.len() {
            return None;
        }

        let valid = (k..attempt.len() - 1).map(|j| (j - k, j)).rev();
        for (x, y) in valid {
            let mut middle = attempt[x..y].to_vec();
            middle.sort_unstable();
            if *middle.as_slice() == attempt[x..y] {
                continue;
            };
            let new = [&attempt[..x], &middle, &attempt[y..]].concat();
            if self.consider(&new) {
                return Some(new);
            }
        }
        None
    }

    fn shrink_swap(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        let valid = (k..attempt.len() - 1).map(|j| (j - k, j)).rev();
        for (x, y) in valid {
            if attempt[x] == attempt[y] {
                continue;
            }
            let mut new = attempt.to_owned();
            // Swap
            new[x] = attempt[y];
            new[y] = attempt[x];
            // For now use inefficient reducing algorithm to get it out.
            match self.shrink_reduce(&new) {
                Some(result) => return Some(result),
                None => continue,
            }
        }
        None
    }

    fn shrink_redistribute(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if attempt.len() < k { 
            return None;
        }

        let mut new = attempt.to_owned();
        let valid = (0..attempt.len()-k).map(|j| (j, j+k));
        for (x, y) in valid {
            if attempt[x] == 0 {
                continue;
            }
            let redistribute = |mut new_: Vec<u64>, v| {
                new_[x] = v;
                new_[y] = attempt[x] + attempt[y] - v;
                new_
            };

            let mut low = 0;
            let mut high = attempt[x];

            new = redistribute(new, low);
            if self.consider(&new) {
                return Some(new);
            }

            while low+1 < high {
                let mid  = low + (high - low) / 2;
                new = redistribute(new, mid);
                if self.consider(&new) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            new = redistribute(new, high);
        }
        if new == attempt {
            None
        } else {
            Some(new)
        }
    }

    fn shrink(&mut self) {

        if let Some(data) = &self.result {
            let result = data.clone();
            let mut attempt = result;
            let mut improved = true;
            while improved {
                improved = false;

                // Deleting choices we made in chunks
                for k in &[8, 4, 2, 1] {
                    while let Some(new) = self.shrink_remove(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                // Replacing blocks by zeroes
                // We *do* use length one here to avoid special casing in the
                // binary search algorithm.
                for k in &[8, 4, 2, 1] {
                    while let Some(new) = self.shrink_zeroes(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                // Replace individual numbers by lower numbers
                if let Some(new) = self.shrink_reduce(&attempt) {
                    attempt = new;
                    improved = true;
                }

                for k in &[8, 4, 2] {
                    while let Some(new) = self.shrink_sort(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                for k in &[2, 1] {
                    while let Some(new) = self.shrink_swap(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                for k in &[2, 1] {
                    while let Some(new) = self.shrink_redistribute(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                if !improved {
                    println!("not improved, exiting, {:?}", attempt);
                };
            }
            self.result = Some(attempt);
        }
    }
}

fn example_test(tc: &mut TestCase) -> Status {
    let ls = data::vectors(data::integers(95, 105), 9, 11).produce(tc);
    match ls {
        Ok(list) => if list.iter().sum::<i64>() > 1000 { Status::Interesting } else { Status::Valid },
        Err(_) => { Status::Invalid }
    }
}

fn main() {
    let mut ts = TestState::new(thread_rng(), Box::new(example_test), 10);
    let db_ = database::DirectoryBasedExampleDatabase::new(".minithesis-db");
    ts.run();
    println!("Test result {:?}", ts.result);
}

use std::panic;


mod tests {
    use super::*;

    #[test]
    fn test_function_interesting() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Status::Interesting), 10000);

        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(ts.test_function(&mut tc));
        assert_eq!(ts.result, Some(vec![]));

        ts.result = Some(vec![1, 2, 3, 4]);
        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(ts.test_function(&mut tc));
        assert_eq!(ts.result, Some(vec![]));


        let mut tc = TestCase::new(vec![1, 2, 3, 4], thread_rng(), 10000);
        assert!(!ts.test_function(&mut tc));
        assert_eq!(ts.result, Some(vec![]));
    }

    #[test]
    fn test_function_valid() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Status::Valid), 10000);

        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(!ts.test_function(&mut tc));
        assert_eq!(ts.result, None);

        ts.result = Some(vec![1, 2, 3, 4]);
        ts.test_function(&mut TestCase::new((&[]).to_vec(), thread_rng(), 10000));
        assert_eq!(ts.result, Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_function_invalid() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Status::Invalid), 10000);

        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(!ts.test_function(&mut tc));
        assert_eq!(ts.result, None);
    }
    
    #[test]
    fn shrink_remove() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Status::Interesting), 10000);
        ts.result = Some(vec![1, 2, 3]);

        assert_eq!(ts.shrink_remove(&[1, 2], 1), Some(vec![1]));
        assert_eq!(ts.shrink_remove(&[1, 2], 2), Some(vec![]));

        ts.result = Some(vec![1, 2, 3, 4, 5]);
        assert_eq!(ts.shrink_remove(&[1, 2, 3, 4], 2), Some(vec![1, 2]));

        // Slightly complex case to make sure it doesn't only check the last ones.
        fn second_is_five(tc: &mut TestCase) -> Status {
            let ls = (0..3).map(|_| tc.choice(10).unwrap()).collect::<Vec<_>>();
            if ls[2] == 5 { Status::Interesting } else { Status::Valid }
        }
        let mut ts = TestState::new(thread_rng(), Box::new(second_is_five), 10000);
        ts.result = Some(vec![1, 2, 5, 4, 5]);
        assert_eq!(ts.shrink_remove(&[1, 2, 5, 4, 5], 2), Some(vec![1, 2, 5]));
    }

    #[test]
    fn shrink_redistribute() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Status::Interesting), 10000);

        ts.result = Some(vec![500, 500, 500, 500]);
        assert_eq!(ts.shrink_redistribute(&[500, 500], 1), Some(vec![0, 1000]));

        ts.result = Some(vec![500, 500, 500, 500]);
        assert_eq!(ts.shrink_redistribute(&[500, 500, 500], 2), Some(vec![0, 500, 1000]));
    }

    #[test]
    fn produce_with_deterministic_test_case() {
        // TODO: add more tests here
        let d = data::vectors(data::integers(0, 10000), 0, 1000);
        let mut tc = TestCase::for_choices(vec![1, 1001, 0]);
        assert_eq!(d.produce(&mut tc).unwrap(), vec![1001]);
    }

    #[test]
    fn finds_small_list() {
        fn sum_greater_1000(tc: &mut TestCase) -> Status {
            let d = data::vectors(data::integers(0, 10000), 0, 1000);
            match d.produce(tc) {
                Ok(ls) => if ls.iter().sum::<i64>() > 1000 { Status::Interesting } else { Status::Valid },
                Err(_) => Status::Invalid
            }
        }
        

        let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
        ts.run();

        let d = data::vectors(data::integers(0, 10000), 0, 1000);
        assert_eq!(ts.result_as(d).unwrap(), vec![1001]);
    }

    #[test]
    fn finds_small_list_even_with_bad_lists() {
        use std::convert::TryInto;

        struct BadList;
        impl Possibility<Vec<i64>> for BadList {
            fn produce(&self, tc: &mut TestCase) -> Result<Vec<i64>, Error> {
                let n = tc.choice(10)?;
                let result = (0..n).map(|_| {tc.choice(10000)}).collect::<Result<Vec<u64>, Error>>()?;
                Ok(result.iter().map(|i| *i as i64).collect())
            }
        }

        fn sum_greater_1000(tc: &mut TestCase) -> Status {
            let ls = BadList.produce(tc);
            match BadList.produce(tc) {
                Ok(ls) => if ls.iter().sum::<i64>() > 1000 { Status::Interesting } else { Status::Valid },
                Err(_) => Status::Invalid
            }
        }
    }
}
