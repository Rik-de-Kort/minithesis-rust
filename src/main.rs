use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Debug)]
enum MTErr {
    Frozen,
    StopTest,
}

#[derive(Debug, PartialEq)]
enum MTStatus {
    Overrun,
    Invalid,
    Valid,
    Interesting,
}

type RandomMethod = Box<dyn Fn(&mut ChaCha8Rng) -> u64>;
struct TestCase {
    prefix: Vec<u64>,
    random: ChaCha8Rng,
    max_size: usize,
    choices: Vec<u64>,
    status: Option<MTStatus>,
    // print_results: bool,
    depth: u64,
    // target_score: f64
}

impl TestCase {
    fn new() -> TestCase {
        TestCase {
            prefix: vec![],
            random: ChaCha8Rng::seed_from_u64(0), // Todo: fix up proper randomness
            max_size: 10000,
            choices: vec![],
            status: None,
            depth: 0,
        }
    }

    /// Return an integer in the range [0, n]
    fn choice(&mut self, n: u64) -> Result<u64, MTErr> {
        self.make_choice(n, Box::new(move |rng| rng.gen_range(0, n + 1)))
    }

    /// Return 1 with probability p, 0 otherwise.
    fn weighted(&mut self, p: f64) -> Result<u64, MTErr> {
        match self.random.gen_bool(p) {
            true => {
                self.choices.push(1);
                Ok(1)
            }
            false => {
                self.choices.push(0);
                Ok(0)
            }
        }
    }

    /// Insert a fake choice in the choice sequence
    fn forced_choice(&mut self, n: u64) -> Result<u64, MTErr> {
        match self.status {
            None => {
                if self.choices.len() >= self.max_size {
                    self.mark_status(MTStatus::Overrun)
                } else {
                    self.choices.push(n);
                    Ok(n)
                }
            }
            Some(_) => Err(MTErr::Frozen),
        }
    }

    /// Mark this test case as invalid
    fn reject(&mut self) -> Result<u64, MTErr> {
        self.mark_status(MTStatus::Invalid)
    }

    /// Return a possible value
    fn any<T>(&mut self, p: &Possibility<T>) -> Result<T, MTErr> {
        match p(self) {
            Ok(val) => {
                self.depth += 1;
                Ok(val)
            }
            Err(e) => Err(e),
        }
    }

    // Note that mark_status never returns u64
    fn mark_status(&mut self, status: MTStatus) -> Result<u64, MTErr> {
        match self.status {
            None => {
                self.status = Some(status);
                Err(MTErr::StopTest)
            }
            Some(_) => Err(MTErr::Frozen),
        }
    }

    fn make_choice(&mut self, n: u64, rnd_method: RandomMethod) -> Result<u64, MTErr> {
        match self.status {
            None => {
                if self.choices.len() >= self.max_size {
                    self.mark_status(MTStatus::Overrun)
                } else if self.choices.len() < self.prefix.len() {
                    let result = self.prefix[self.choices.len()];
                    self.choices.push(result);
                    Ok(result)
                } else {
                    let result = rnd_method(&mut self.random);
                    if result > n {
                        self.mark_status(MTStatus::Invalid)
                    } else {
                        self.choices.push(result);
                        Ok(result)
                    }
                }
            }
            Some(_) => Err(MTErr::Frozen),
        }
    }
}

/// Represents some range of values that might be used in a test, that can be requested from a
/// TestCase.
///
/// Pass one of these to TestCase.any to get a concrete value
type Possibility<T> = Box<dyn Fn(&mut TestCase) -> Result<T, MTErr>>;

use std::convert::TryInto;

fn integers(m: i64, n: i64) -> Possibility<i64> {
    let t: u64 = (n - m).try_into().unwrap();
    let produce = move |tc: &mut TestCase| {
        let c: i64 = tc.choice(t)?.try_into().unwrap();
        Ok(m + c)
    };
    Box::new(produce)
}

fn vecs<T: 'static + std::fmt::Debug>(
    elements: Possibility<T>,
    min_size: usize,
    max_size: usize,
) -> Possibility<Vec<T>> {
    let produce = move |tc: &mut TestCase| {
        let mut result = vec![];
        loop {
            if result.len() < min_size {
                tc.forced_choice(1)?;
            } else if result.len() + 1 >= max_size {
                tc.forced_choice(0)?;
                break;
            } else if tc.weighted(0.9)? == 0 {
                break;
            }
            result.push(tc.any(&elements)?);
        }
        Ok(result)
    };
    Box::new(produce)
}

/// For this we define a test as a routine returning true or false.
type PropertyTest<T> = Box<dyn Fn(&mut T) -> Result<u64, MTErr>>;

enum TestResult<T> {
    Pass,
    Unsatisfiable,
    Failing(T),
}

fn run_test<T>(
    test: Box<dyn Fn(&mut TestCase) -> bool>,
    max_examples: usize,
    random: ChaCha8Rng,
) -> TestResult<T> {
    let mark_failures_interesting = move |tc: &mut TestCase| {
        if !test(tc) {
            tc.mark_status(MTStatus::Interesting)
        } else {
            Ok(0)
        }
    };

    let ts = TestingState {
        random: random,
        test_function: Box::new(mark_failures_interesting),
        max_examples: max_examples,
        valid_test_cases: 0,
        calls: 0,
        result: None,
        test_is_trivial: false,
    };

    TestResult::<T>::Pass
}

fn example_test(tc: &mut TestCase) -> bool {
    let ls: Vec<i64> = tc.any(&vecs(integers(0, 100), 5, 10)).unwrap();
    ls.iter().sum::<i64>() < 1000
}

fn sort_key<T>(x: &T) -> u64 {
    0
}

struct TestingState {
    random: ChaCha8Rng,
    max_examples: usize,
    test_function: PropertyTest<TestCase>,
    valid_test_cases: usize,
    calls: usize,
    result: Option<Vec<u64>>,
    test_is_trivial: bool,
}

impl TestingState {
    fn test_function(&mut self, mut test_case: TestCase) -> Result<bool, MTErr> {
        match (self.test_function)(&mut test_case) {
            Ok(_) => {}
            Err(_) => {}
        }

        if test_case.status == None {
            test_case.status = Some(MTStatus::Valid)
        }
        self.calls += 1;

        let status = test_case.status.unwrap();

        if (status == MTStatus::Invalid
            || status == MTStatus::Valid
            || status == MTStatus::Interesting)
            && test_case.choices.len() == 0
        {
            self.test_is_trivial = true;
        }
        if (status == MTStatus::Valid || status == MTStatus::Interesting) {
            self.valid_test_cases += 1;

            // Todo: implement targeting
        }
        if status == MTStatus::Interesting {
            if self.result == None
                || sort_key(self.result.as_ref().unwrap()) > sort_key(&test_case.choices)
            {
                self.result = Some(test_case.choices)
            }
        }

        Ok(true)
    }

    // NEXT TODO: implement run() to finish run_test for example test!
}

fn main() {
    // run_test(Box::new(example), 10, ChaCha8Rng::seed_from_u64(0));
    let mut tc = TestCase::new();
    let p = vecs(integers(0, 10), 10, 10);
    println!("Selected integer: {:?}", tc.any(&p));
    let r: u64 = (vec![1, 2, 3]).iter().sum();
}
