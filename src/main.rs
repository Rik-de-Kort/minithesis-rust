use rand::prelude::*;

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

type RandomMethod = Box<dyn Fn(&mut ThreadRng) -> u64>;
struct TestCase {
    prefix: Vec<u64>,
    random: ThreadRng,
    max_size: usize,
    choices: Vec<u64>,
    status: Option<MTStatus>,
    // print_results: bool,
    depth: u64,
}

impl TestCase {
    fn new(prefix: Vec<u64>, random: ThreadRng, max_size: usize) -> TestCase {
        TestCase {
            prefix: prefix,
            random: random,
            max_size: max_size,
            choices: vec![],
            status: None,
            depth: 0,
        }
    }

    /// Insert a definite choice in the choice sequence
    /// N.B. All integrity checks happen here!
    fn det_choice(&mut self, n: u64) -> Result<u64, MTErr> {
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
    fn weighted(&mut self, p: f64) -> Result<u64, MTErr> {
        if self.random.gen_bool(p) {
            self.det_choice(1)
        } else {
            self.det_choice(0)
        }
    }

    /// Mark this test case as invalid
    fn reject(&mut self) -> MTErr {
        self.mark_status(MTStatus::Invalid)
    }

    /// Return a possible value
    fn any<T>(&mut self, p: &impl Possibility<T>) -> Result<T, MTErr> {
        match p.produce(self) {
            Ok(val) => {
                self.depth += 1;
                Ok(val)
            }
            Err(e) => Err(e),
        }
    }

    // Note that mark_status never returns u64
    fn mark_status(&mut self, status: MTStatus) -> MTErr {
        match self.status {
            Some(_) => MTErr::Frozen,
            None => {
                self.status = Some(status);
                MTErr::StopTest
            }
        }
    }

    // Return an integer in the range [0, n]
    fn choice(&mut self, n: u64) -> Result<u64, MTErr> {
        if self.choices.len() < self.prefix.len() {
            self.det_choice(self.prefix[self.choices.len()])
        } else {
            let result = self.random.gen_range(0, n + 1);
            self.det_choice(result)
        }
    }
}

/// Represents some range of values that might be used in a test, that can be requested from a
/// TestCase.
trait Possibility<T> {
    fn produce(&self, tc: &mut TestCase) -> Result<T, MTErr>;
}

struct Vectors<U, T: Possibility<U>> {
    elements: T,
    min_size: usize,
    max_size: usize,
    phantom: std::marker::PhantomData<U>, // Required for type check
}

impl<U, T: Possibility<U>> Vectors<U, T> {
    fn new(elements: T, min_size: usize, max_size: usize) -> Self {
        Vectors {
            elements: elements,
            min_size: min_size,
            max_size: max_size,
            phantom: std::marker::PhantomData,
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

struct Integers {
    minimum: i64,
    maximum: i64,
    range: u64,
}

use std::convert::TryInto;
impl Integers {
    fn new(minimum: i64, maximum: i64) -> Self {
        Integers {
            minimum: minimum,
            maximum: maximum,
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

/// We are seeing if things are interesting (true or false)
type InterestingTest<T> = Box<dyn Fn(&mut T) -> bool>;

struct TestingState {
    random: ThreadRng,
    max_examples: usize,
    is_interesting: InterestingTest<TestCase>,
    valid_test_cases: usize,
    calls: usize,
    result: Option<Vec<u64>>,
    best_scoring: Option<Vec<u64>>, // Todo: check type
    test_is_trivial: bool,
}

impl TestingState {
    fn test_function(&mut self, mut test_case: TestCase) {
        if (self.is_interesting)(&mut test_case) {
            test_case.status = Some(MTStatus::Interesting);
        } else if test_case.status == None {
            test_case.status = Some(MTStatus::Valid)
        }

        self.calls += 1;

        match test_case.status {
            None => panic!("Didn't expect test case status to be empty!"),
            Some(MTStatus::Invalid) => {
                self.test_is_trivial = test_case.choices.len() == 0;
            }
            Some(MTStatus::Valid) => {
                self.test_is_trivial = test_case.choices.len() == 0;
                self.valid_test_cases += 1;
            }
            Some(MTStatus::Interesting) => {
                self.test_is_trivial = test_case.choices.len() == 0;
                self.valid_test_cases += 1;

                if self.result == None || self.result.as_ref().unwrap() > &test_case.choices {
                    self.result = Some(test_case.choices)
                }
            }
            Some(MTStatus::Overrun) => {
                panic!("Test case length overrun!");
            }
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
            self.test_function(TestCase::new(vec![], self.random.clone(), 8 * 1024));
        }
    }

    fn shrink(&mut self) {
        if self.result == None {
            return;
        }
    }
}

/// Note that we invert the relationship: true means "interesting"
fn example_test(tc: &mut TestCase) -> bool {
    let ls: Vec<i64> = tc
        .any(&Vectors::new(Integers::new(95, 105), 9, 11))
        .unwrap();
    println!("running with list {:?}", ls);
    ls.iter().sum::<i64>() > 1000
}

fn main() {
    let mut ts = TestingState {
        random: thread_rng(),
        is_interesting: Box::new(example_test),
        max_examples: 10,
        valid_test_cases: 0,
        calls: 0,
        result: None,
        best_scoring: None,
        test_is_trivial: false,
    };
    ts.run();
    println!("Test result {:?}", ts.result);
}
