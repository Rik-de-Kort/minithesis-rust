use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::error::Error;

#[derive(Debug)]
struct Frozen;

impl Error for Frozen {}

impl std::fmt::Display for Frozen {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Test case is frozen!")
    }
}

#[derive(Debug)]
struct StopTest;

impl Error for StopTest {}

impl std::fmt::Display for StopTest {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Stop test early")
    }
}

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
            random: ChaCha8Rng::seed_from_u64(0),  // Todo: fix up proper randomness
            max_size: 10000,
            choices: vec![],
            status: None,
            depth: 0
        }
    }

    /// Return an integer in the range [0, n]
    fn choice(&mut self, n: u64) -> Result<u64, MTErr> {
        self.make_choice(n, Box::new(move |rng| rng.gen_range(0, n+1)))
    }

    /// Return 1 with probability p, 0 otherwise. 
    fn weighted(&mut self, p: f64) -> Result<u64, MTErr> {
        match self.random.gen_bool(p) {
            true => { 
                self.choices.push(1);
                Ok(1)
            },
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
            },
            Some(_) => Err(MTErr::Frozen)
        }

    }

    /// Mark this test case as invalid
    fn reject(&mut self) -> Result<u64, MTErr> {
        self.mark_status(MTStatus::Invalid)
    }

    /// Return a possible value
    fn any<T>(&mut self, p: Possibility<T>) -> Result<T, MTErr> {
        match p(self) {
            Ok(val) => {
                self.depth += 1;
                Ok(val)
            },
            Err(e) => Err(e)
        }


    }

    // Note that mark_status never returns u64
    fn mark_status(&mut self, status: MTStatus) -> Result<u64, MTErr> {
        match self.status {
            None => {
                self.status = Some(status);
                Err(MTErr::StopTest)
            },
            Some(_) => {
                Err(MTErr::Frozen)
            }
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
            },
            Some(_) => Err(MTErr::Frozen)
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
    let t: u64 = (n-m).try_into().unwrap();
    let result = move |tc: &mut TestCase| {
            let c: i64 = tc.choice(t)?.try_into().unwrap();
            Ok(m+c)
        };
    Box::new(result)
}


/// For this we define a test as a routine returning true or false. 
type PropertyTest<T> = Box<dyn Fn(T) -> bool>;

enum TestResult<T> {
    Pass,
    Unsatisfiable,
    Failing(T)
}

fn run_test<T>(test: PropertyTest<T>, max_examples: u64, random: ChaCha8Rng) -> TestResult<T> {
    TestResult::<T>::Pass
}

fn example(x: i64) -> bool {
    if x % 2 == 0 { false } else { true }
}

fn sort_key<T>(x: T) -> u64 { 0 }

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
    fn test_function(&mut self, test_case: TestCase) -> Result<bool, MTErr> {
        match (self.test_function)(test_case) {
            true => {}
            false => {}
        }

        if test_case.status == None {
            test_case.status = Some(Status::Valid)
        }
        self.calls += 1;

        let status = test_case.status.unwrap();

        if (status == MTStatus.Invalid || status == MTStatus.Valid || status == MTStatus.Interesting) && test_case.choices.len() == 0 {
            self.test_is_trivial = true;
        }
        if (status == MTStatus.Valid || status == MTStatus.Interesting) {
            self.valid_test_cases += 1;
            
            // Todo: implement targeting
        }
        if test_case.status == MTStatus.Interesting {
            if self.result == None or sort_key(self.result.unwrap()) > sort_key(test_case.choices) {
                self.result = Some(test_case.choices)
            }
        }

        Ok(true)
}


fn main() {
    fn foo() -> bool {true};
    println!("Hello, world! {}", foo());
    run_test(Box::new(example), 10, ChaCha8Rng::seed_from_u64(0));
    let mut tc = TestCase::new();
    let p = integers(0, 10);
    println!("Selected integer: {}", tc.any(p).unwrap());
}
