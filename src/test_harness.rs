use crate::data::Possibility;
use rand::prelude::*;

#[derive(Debug)]
pub enum MTErr {
    Frozen,
    StopTest,
}

#[derive(Debug, PartialEq)]
pub enum MTStatus {
    Overrun,
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
    status: Option<MTStatus>,
    // print_results: bool,
    depth: u64,
}

impl TestCase {
    pub fn new(prefix: &[u64], random: ThreadRng, max_size: usize) -> TestCase {
        TestCase {
            prefix: prefix.to_vec(),
            random,
            max_size,
            choices: vec![],
            status: None,
            depth: 0,
        }
    }

    pub fn for_choices(prefix: &[u64]) -> TestCase {
        TestCase {
            max_size: prefix.len(),
            prefix: prefix.to_vec(),
            random: thread_rng(),
            choices: vec![],
            status: None,
            depth: 0,
        }
    }

    /// Insert a definite choice in the choice sequence
    /// N.B. All integrity checks happen here!
    pub fn det_choice(&mut self, n: u64) -> Result<u64, MTErr> {
        match self.status {
            None => {
                if self.choices.len() >= self.max_size {
                    Err(self.mark_status(MTStatus::Overrun))
                } else {
                    self.choices.push(n);
                    Ok(n)
                }
            }
            Some(_) => Err(MTErr::Frozen),
        }
    }

    /// Return 1 with probability p, 0 otherwise.
    pub fn weighted(&mut self, p: f64) -> Result<u64, MTErr> {
        if self.random.gen_bool(p) {
            self.det_choice(1)
        } else {
            self.det_choice(0)
        }
    }

    /// Mark this test case as invalid
    pub fn reject(&mut self) -> MTErr {
        self.mark_status(MTStatus::Invalid)
    }

    /// If this precondition is not met, abort the test and mark this test case as invalid
    pub fn assume(&mut self, precondition: bool) -> Option<MTErr> {
        if !precondition {
            Some(self.reject())
        } else {
            None
        }
    }

    /// Return a possible value
    pub fn any<T>(&mut self, p: &impl Possibility<T>) -> Result<T, MTErr> {
        self.depth += 1;
        let result = p.produce(self);
        self.depth -= 1;
        result
    }

    // Note that mark_status never returns u64
    pub fn mark_status(&mut self, status: MTStatus) -> MTErr {
        match self.status {
            Some(_) => MTErr::Frozen,
            None => {
                self.status = Some(status);
                MTErr::StopTest
            }
        }
    }

    /// Return an integer in the range [0, n]
    pub fn choice(&mut self, n: u64) -> Result<u64, MTErr> {
        if self.choices.len() < self.prefix.len() {
            self.det_choice(self.prefix[self.choices.len()])
        } else {
            let result = self.random.gen_range(0, n + 1);
            self.det_choice(result)
        }
    }
}

type InterestingTest<T> = Box<dyn Fn(&mut T) -> bool>;

pub struct TestState {
    random: ThreadRng,
    max_examples: usize,
    is_interesting: InterestingTest<TestCase>,
    valid_test_cases: usize,
    calls: usize,
    pub result: Option<Vec<u64>>,
    best_scoring: Option<Vec<u64>>, // Todo: check type
    test_is_trivial: bool,
}

const BUFFER_SIZE: usize = 8 * 1024;

impl TestState {
    pub fn new(
        random: ThreadRng,
        test_function: InterestingTest<TestCase>,
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

    /// Function to run tests, returns true if result was interesting
    fn test_function(&mut self, mut test_case: &mut TestCase) -> bool {
        if (self.is_interesting)(&mut test_case) {
            test_case.status = Some(MTStatus::Interesting);
        } else if test_case.status == None {
            test_case.status = Some(MTStatus::Valid)
        }

        self.calls += 1;

        match test_case.status {
            None => unreachable!("Didn't expect test case status to be empty!"),
            Some(MTStatus::Invalid) => {
                self.test_is_trivial = test_case.choices.is_empty();
                false
            }
            Some(MTStatus::Valid) => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;
                false
            }
            Some(MTStatus::Interesting) => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;

                if self.result == None || (*self.result.as_ref().unwrap() > test_case.choices) {
                    self.result = Some(test_case.choices.clone());
                }
                true
            }
            Some(MTStatus::Overrun) => false,
        }
    }

    pub fn run(&mut self) {
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
            self.test_function(&mut TestCase::new(&[], self.random, BUFFER_SIZE));
        }
    }

    fn shrink_remove(&mut self, attempt: &[u64], k: usize) -> Option<Vec<u64>> {
        if k > attempt.len() {
            return None;
        }

        // Generate all valid removals (don't worry, it's lazy!)
        let valid = (k..attempt.len() - 1).map(|j| (j - k, j)).rev();
        for (x, y) in valid {
            let mut new = [&attempt[..x], &attempt[y..]].concat();

            if self.test_function(&mut TestCase::for_choices(&new)) {
                return Some(new);
            } else if x > 0 && new[x - 1] > 0 {
                // Short-circuit prevents overflow
                new[x - 1] -= 1;
                if self.test_function(&mut TestCase::for_choices(&new)) {
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
            if self.test_function(&mut TestCase::for_choices(&new)) {
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
                if self.test_function(&mut TestCase::for_choices(&new)) {
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
            if self.test_function(&mut TestCase::for_choices(&new)) {
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
            }
            self.result = Some(attempt);
        }
    }
}

