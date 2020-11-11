use rand::prelude::*;
mod database;

#[derive(Debug)]
pub enum Error {
    Overrun,
    Invalid,
}

#[derive(Debug)]
pub struct TestCase {
    prefix: Vec<u64>,
    random: ThreadRng,
    max_size: usize,
    choices: Vec<u64>,
    targeting_score: Option<f64>,
}

use crate::data::Possibility;
impl TestCase {
    fn new(prefix: Vec<u64>, random: ThreadRng, max_size: usize) -> TestCase {
        TestCase {
            prefix,
            random,
            max_size,
            choices: vec![],
            targeting_score: None,
        }
    }

    fn for_choices(prefix: Vec<u64>) -> TestCase {
        TestCase {
            max_size: prefix.len(),
            prefix,
            random: thread_rng(),
            choices: vec![],
            targeting_score: None,
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
    fn assume(&mut self, precondition: bool) -> Result<(), Error> {
        if !precondition {
            Err(Error::Invalid)
        } else {
            Ok(())
        }
    }

    /// Return an integer in the range [0, n]
    fn choice(&mut self, n: u64) -> Result<u64, Error> {
        if self.choices.len() < self.prefix.len() {
            let preordained = self.prefix[self.choices.len()];
            if preordained > n {
                Err(Error::Invalid)
            } else {
                self.forced_choice(self.prefix[self.choices.len()])
            }
        } else {
            let result = self.random.gen_range(0, n + 1);
            self.forced_choice(result)
        }
    }

    /// Add a score to target. Put an expression here!
    /// If called more than one
    fn target(&mut self, score: f64) {
        if self.targeting_score != None {
            println!(
                "TestCase::target called twice on test case object {:?}. Overwriting score.",
                self
            );
        }
        self.targeting_score = Some(score);
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
    impl<T, U, F: Fn(T) -> Q, P: Possibility<T>, Q: Possibility<U>> Possibility<U>
        for Bind<T, U, F, P, Q>
    {
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

    pub struct Pairs<U, T: Possibility<U>, V, S: Possibility<V>> {
        first: T,
        second: S,
        phantom_u: PhantomData<U>,
        phantom_v: PhantomData<V>,
    }

    impl<U, T: Possibility<U>, V, S: Possibility<V>> Pairs<U, T, V, S> {
        pub fn new(first: T, second: S) -> Pairs<U, T, V, S> {
            Pairs {
                first,
                second,
                phantom_u: PhantomData,
                phantom_v: PhantomData,
            }
        }
    }

    impl<U, T: Possibility<U>, V, S: Possibility<V>> Possibility<(U, V)> for Pairs<U, T, V, S> {
        fn produce(&self, tc: &mut TestCase) -> Result<(U, V), Error> {
            Ok((self.first.produce(tc)?, self.second.produce(tc)?))
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

    pub fn pairs<U, T: Possibility<U>, V, S: Possibility<V>>(
        first: T,
        second: S,
    ) -> Pairs<U, T, V, S> {
        Pairs::new(first, second)
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
    is_interesting: Box<dyn Fn(&mut TestCase) -> Result<bool, Error>>,
    valid_test_cases: usize,
    calls: usize,
    result: Option<Vec<u64>>,
    best_scoring: Option<(f64, Vec<u64>)>,
    test_is_trivial: bool,
}

const BUFFER_SIZE: usize = 8 * 1024;

impl TestState {
    pub fn new(
        random: ThreadRng,
        test_function: Box<dyn Fn(&mut TestCase) -> Result<bool, Error>>,
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

    fn update_target(&mut self, test_case: &TestCase) -> bool {
        if let Some(score) = test_case.targeting_score {
            if self.best_scoring == None || self.best_scoring.as_ref().unwrap().0 < score {
                self.best_scoring = Some((score, test_case.choices.clone()));
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    // This is a new variant of test_function which also returns whether or not the
    // test case was better than the previous best scoring one!
    fn test_function(&mut self, test_case: &mut TestCase) -> (bool, bool) {
        self.calls += 1;
        match (self.is_interesting)(test_case) {
            Ok(false) => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;
                (false, self.update_target(test_case))
            }
            Ok(true) => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;

                if self.result == None
                    || self.result.as_ref().unwrap().len() > test_case.choices.len()
                    || *self.result.as_ref().unwrap() > test_case.choices
                {
                    self.result = Some(test_case.choices.clone());
                    (true, self.update_target(test_case))
                } else {
                    (false, self.update_target(test_case))
                }
            }
            Err(_) => (false, false),
        }
    }

    fn adjust(&mut self, attempt: &[u64]) -> bool {
        let result = self.test_function(&mut TestCase::for_choices(attempt.to_owned()));
        result.1
    }

    fn target(&mut self) {
        if self.result != None {
            return;
        }

        if self.best_scoring.is_some() {
            while self.should_keep_generating() {
                // It may happen that choices is all zeroes, and that targeting upwards
                // doesn't do anything. In this case, the loop will run until max_examples
                // is exhausted.

                // Could really use destructuring assignment here...
                let mut new = if let Some((_, choices)) = &self.best_scoring {
                    choices.clone()
                } else {
                    unreachable!()
                };
                let i = self.random.gen_range(0, new.len());

                // Can we climb up?
                new[i] += 1;
                if self.adjust(&new) {
                    let mut k = 1;
                    new[i] += k;
                    let mut res = self.adjust(&new);
                    while self.should_keep_generating() && res {
                        k *= 2;
                        new[i] += k;
                        res = self.adjust(&new);
                    }
                    while k > 0 {
                        while self.should_keep_generating() && self.adjust(&new) {
                            new[i] += k;
                        }
                        k /= 2;
                    }
                }

                // Or should we climb down?
                if new[i] < 1 {
                    continue;
                }
                new[i] -= 1;
                if self.adjust(&new) {
                    let mut k = 1;
                    if new[i] < k {
                        continue;
                    }
                    new[i] -= k;

                    while self.should_keep_generating() && self.adjust(&new) {
                        if new[i] < k {
                            break;
                        }
                        new[i] -= k;
                        k *= 2;
                    }
                    while k > 0 {
                        while self.should_keep_generating() && self.adjust(&new) {
                            if new[i] < k {
                                break;
                            }
                            new[i] -= k;
                        }
                        k /= 2;
                    }
                }
            }
        }
    }

    fn run(&mut self) {
        self.generate();
        self.target();
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
        if Some(choices) == self.result.as_deref() {
            true
        } else {
            self.test_function(&mut TestCase::for_choices(choices.to_vec()))
                .0
        }
    }

    fn shrink_remove(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if k > attempt.len() {
            return None;
        }

        // Generate all valid removals (don't worry, it's lazy!)
        let valid = (0..=attempt.len() - k).map(|j| (j, j + k)).rev();
        for (x, y) in valid {
            let head = &attempt[..x];
            let tail = if y < attempt.len() {
                &attempt[y..]
            } else {
                &[]
            };
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
            if let Some(x) = bin_search_down(0, new[i], &mut |n| {
                new[i] = n;
                self.consider(&new)
            }) {
                new[i] = x
            }
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

            // We're swapping x and y, but also immediately reducing x.
            new[y] = attempt[x];
            if let Some(i) = bin_search_down(0, attempt[y], &mut |n| {
                new[x] = n;
                self.consider(&new)
            }) {
                new[x] = i;
                return Some(new);
            } else {
                continue;
            }
        }
        None
    }

    fn shrink_redistribute(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if attempt.len() < k {
            return None;
        }

        let mut new = attempt.to_owned();
        let valid = (0..attempt.len() - k).map(|j| (j, j + k));
        for (x, y) in valid {
            if attempt[x] == 0 {
                continue;
            }

            if let Some(v) = bin_search_down(0, attempt[x], &mut |n| {
                new[x] = n;
                new[y] = attempt[x] + attempt[y] - n;
                self.consider(&new)
            }) {
                new[x] = v;
                new[y] = attempt[x] + attempt[y] - v;
            }
        }
        if new == attempt {
            None
        } else {
            Some(new)
        }
    }

    fn shrink(&mut self) {
        if let Some(data) = &self.result {
            let mut attempt = data.clone();
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
                for k in &[8, 4, 2] {
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

                // Sort sublists
                for k in &[8, 4, 2] {
                    while let Some(new) = self.shrink_sort(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                // Swap numbers distance k apart, and shrink the first one.
                for k in &[2, 1] {
                    while let Some(new) = self.shrink_swap(&attempt, *k) {
                        attempt = new;
                        improved = true;
                    }
                }

                // Redistribute values between nearby numbers
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
        }
    }
}

fn bin_search_down(mut low: u64, mut high: u64, p: &mut dyn FnMut(u64) -> bool) -> Option<u64> {
    if p(low) {
        return Some(low);
    }
    if !p(high) {
        return None;
    }

    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if p(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    Some(high)
}

fn example_test(tc: &mut TestCase) -> Result<bool, Error> {
    let ls = data::vectors(data::integers(95, 105), 9, 11).produce(tc)?;
    tc.target(ls[0] as f64);
    Ok(ls.iter().sum::<i64>() > 1000)
}

fn main() {
    fn test(tc: &mut TestCase) -> Result<bool, Error> {
        let n = tc.choice(1001)? as f64;
        let m = tc.choice(1001)? as f64;
        let score = n + m;
        tc.target(score);
        Ok(score >= 2000.0)
    }
    let mut ts = TestState::new(thread_rng(), Box::new(test), 1000);
    ts.run();
    assert!(ts.result.is_some());
}

mod tests {
    use super::*;

    #[test]
    fn test_function_interesting() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Ok(true)), 10000);

        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(ts.test_function(&mut tc).0);
        assert_eq!(ts.result, Some(vec![]));

        ts.result = Some(vec![1, 2, 3, 4]);
        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(ts.test_function(&mut tc).0);
        assert_eq!(ts.result, Some(vec![]));

        let mut tc = TestCase::new(vec![1, 2, 3, 4], thread_rng(), 10000);
        assert!(!ts.test_function(&mut tc).0);
        assert_eq!(ts.result, Some(vec![]));
    }

    #[test]
    fn test_function_valid() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Ok(false)), 10000);

        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(!ts.test_function(&mut tc).0);
        assert_eq!(ts.result, None);

        ts.result = Some(vec![1, 2, 3, 4]);
        ts.test_function(&mut TestCase::new((&[]).to_vec(), thread_rng(), 10000));
        assert_eq!(ts.result, Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_function_invalid() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Err(Error::Invalid)), 10000);

        let mut tc = TestCase::new((&[]).to_vec(), thread_rng(), 10000);
        assert!(!ts.test_function(&mut tc).0);
        assert_eq!(ts.result, None);
    }

    #[test]
    fn shrink_remove() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Ok(true)), 10000);
        ts.result = Some(vec![1, 2, 3]);

        assert_eq!(ts.shrink_remove(&[1, 2], 1), Some(vec![1]));
        assert_eq!(ts.shrink_remove(&[1, 2], 2), Some(vec![]));

        ts.result = Some(vec![1, 2, 3, 4, 5]);
        assert_eq!(ts.shrink_remove(&[1, 2, 3, 4], 2), Some(vec![1, 2]));

        // Slightly complex case to make sure it doesn't only check the last ones.
        fn second_is_five(tc: &mut TestCase) -> Result<bool, Error> {
            let ls = (0..3).map(|_| tc.choice(10).unwrap()).collect::<Vec<_>>();
            Ok(ls[2] == 5)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(second_is_five), 10000);
        ts.result = Some(vec![1, 2, 5, 4, 5]);
        assert_eq!(ts.shrink_remove(&[1, 2, 5, 4, 5], 2), Some(vec![1, 2, 5]));

        fn sum_greater_1000(tc: &mut TestCase) -> Result<bool, Error> {
            let ls = data::vectors(data::integers(0, 10000), 0, 1000).produce(tc)?;
            Ok(ls.iter().sum::<i64>() > 1000)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
        ts.result = Some(vec![1, 10000, 1, 10000]);
        assert_eq!(
            ts.shrink_remove(&[1, 0, 1, 1001, 0], 2),
            Some(vec![1, 1001, 0])
        );

        ts.result = Some(vec![1, 10000, 1, 10000]);
        assert_eq!(ts.shrink_remove(&[1, 0, 1, 1001, 0], 1), None);
    }

    #[test]
    fn shrink_redistribute() {
        let mut ts = TestState::new(thread_rng(), Box::new(|_| Ok(true)), 10000);

        ts.result = Some(vec![500, 500, 500, 500]);
        assert_eq!(ts.shrink_redistribute(&[500, 500], 1), Some(vec![0, 1000]));

        ts.result = Some(vec![500, 500, 500, 500]);
        assert_eq!(
            ts.shrink_redistribute(&[500, 500, 500], 2),
            Some(vec![0, 500, 1000])
        );
    }

    #[test]
    fn finds_small_list() {
        fn sum_greater_1000(tc: &mut TestCase) -> Result<bool, Error> {
            let ls = data::vectors(data::integers(0, 10000), 0, 1000).produce(tc)?;
            Ok(ls.iter().sum::<i64>() > 1000)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
        ts.run();

        assert_eq!(ts.result, Some(vec![1, 1001, 0]));
    }

    #[test]
    fn finds_small_list_debug() {
        fn sum_greater_1000(tc: &mut TestCase) -> Result<bool, Error> {
            let ls = data::vectors(data::integers(0, 10000), 0, 1000).produce(tc)?;
            Ok(ls.iter().sum::<i64>() > 1000)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
        ts.result = Some(vec![1, 0, 1, 1001, 0]);
        // This buggy case came about due to the fact that rust compares vecs element by element.
        // assert!(vec![1, 1001, 0] < vec![1, 0, 1, 1001, 0]);
        assert_eq!(
            ts.shrink_remove(&[1, 0, 1, 1001, 0], 2),
            Some(vec![1, 1001, 0])
        );
        assert_eq!(ts.result, Some(vec![1, 1001, 0]));
    }

    #[test]
    fn finds_small_list_even_with_bad_lists() {
        struct BadList;
        impl Possibility<Vec<i64>> for BadList {
            fn produce(&self, tc: &mut TestCase) -> Result<Vec<i64>, Error> {
                let n = tc.choice(10)?;
                let result = (0..n)
                    .map(|_| tc.choice(10000))
                    .collect::<Result<Vec<u64>, Error>>()?;
                Ok(result.iter().map(|i| *i as i64).collect())
            }
        }

        fn sum_greater_1000(tc: &mut TestCase) -> Result<bool, Error> {
            let ls = BadList.produce(tc)?;
            Ok(ls.iter().sum::<i64>() > 1000)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
        ts.run();
        assert_eq!(ts.result, Some(vec![1, 1001]));
    }

    #[test]
    fn reduces_additive_pairs() {
        fn sum_greater_1000(tc: &mut TestCase) -> Result<bool, Error> {
            let n = tc.choice(1000)?;
            let m = tc.choice(1000)?;
            Ok(m + n > 1000)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
        ts.run();
        assert_eq!(ts.result, Some(vec![1, 1000]));
    }

    #[test]
    fn test_cases_satisfy_preconditions() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let n = tc.choice(10)?;
            tc.assume(n != 0)?;
            Ok(n == 0)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    // TODO: implement Unsatisfiable mechanism
    // TODO: implement caching mechanism

    // Note: cannot reproduce max_examples_is_not_exceeded because no globals
    #[test]
    fn finds_local_maximum() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let m = tc.choice(1000)? as f64;
            let n = tc.choice(1000)? as f64;
            let score = -((m - 500.0).powf(2.0) + (n - 500.0).powf(2.0));
            tc.target(score);
            Ok(m == 500.0 || n == 500.0)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert!(ts.result.is_some());
    }

    #[test] // TODO: this can yield overflow error
    fn can_target_score_upwards_to_interesting() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let n = tc.choice(1000)? as f64;
            let m = tc.choice(1000)? as f64;
            let score = n + m;
            tc.target(score);
            Ok(score >= 2000.0)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 1000);
        ts.run();
        assert!(ts.result.is_some());
    }

    #[test]
    fn can_target_score_upwards_without_failing() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let n = tc.choice(1000)? as f64;
            let m = tc.choice(1000)? as f64;
            let score = n + m;
            tc.target(score);
            Ok(false)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 1000);
        ts.run();
        assert!(ts.result.is_none());
        if let Some((score, _)) = ts.best_scoring {
            assert_eq!(score, 2000.0);
        } else {
            assert!(false, "best scoring not filled")
        }
    }

    // TODO: check frozen
    #[test]
    fn mapped_possibility() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let n = data::integers(0, 5).map(|n| n * 2).produce(tc)?;
            Ok(n % 2 != 0)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn selected_possibility() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let n = data::integers(0, 5)
                .satisfying(|n| n % 2 == 0)
                .produce(tc)?;
            Ok(n % 2 != 0)
        }

        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn bound_possibility() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let t = data::integers(0, 5)
                .bind(|m| data::pairs(data::just(m), data::integers(m, m + 10)))
                .produce(tc)?;
            Ok(t.1 < t.0 || t.0 + 10 < t.1)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn cannot_witness_nothing() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let _ = data::nothing().produce(tc)?;
            Ok(true)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn can_draw_mixture() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let m = data::mix_of(data::integers(-5, 0), data::integers(2, 5)).produce(tc)?;
            Ok(-5 > m || m > 5 || m == 1)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn impossible_weighted() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            for _ in 0..10 {
                if tc.weighted(0.0)? == 1 {
                    assert!(false);
                }
            }
            Ok(false)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn guaranteed_weighted() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            for _ in 0..10 {
                if tc.weighted(1.0)? == 0 {
                    assert!(false);
                }
            }
            Ok(false)
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }

    #[test]
    fn size_bounds_on_vectors() {
        fn test(tc: &mut TestCase) -> Result<bool, Error> {
            let ls = data::vectors(data::integers(0, 10), 1, 3).produce(tc)?;
            Ok(ls.len() < 1 || 3 < ls.len())
        }
        let mut ts = TestState::new(thread_rng(), Box::new(test), 10000);
        ts.run();
        assert_eq!(ts.result, None);
    }
}
