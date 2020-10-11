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
///
/// Pass one of these to TestCase.any to get a concrete value
type Possibility<T> = Box<dyn Fn(&mut TestCase) -> Result<T, MTErr>>;

use std::convert::TryInto;

fn map<T: 'static, U: 'static>(initial: Possibility<T>, f: impl Fn(T)->U + 'static) -> Possibility<U> {
    Box::new(move |tc: &mut TestCase| {
        Ok(f(tc.any(&initial)?))
    })
}

fn bind<T: 'static, U: 'static>(initial: Possibility<T>, f: impl Fn(T)->Possibility<U> + 'static) -> Possibility<U> {
    Box::new(move |tc: &mut TestCase| {
        let mapped = f(tc.any(&initial)?);
        Ok(tc.any(&mapped)?)
    })
}

fn satisfying<T: 'static>(initial: Possibility<T>, f: impl Fn(&T)->bool + 'static) -> Possibility<T> {
    Box::new(move |tc: &mut TestCase| {
        for _ in (0..3) {
            let candidate = tc.any(&initial)?;
            if f(&candidate) { return Ok(candidate); }
        }
        Err(tc.reject())
    })
}

fn integers(m: i64, n: i64) -> Possibility<i64> {
    let t: u64 = (n - m).try_into().unwrap();
    let produce = move |tc: &mut TestCase| {
        let c: i64 = tc.choice(t)?.try_into().unwrap();
        Ok(m + c)
    };
    Box::new(produce)
}

fn vecs<T: 'static>(
    elements: Possibility<T>,
    min_size: usize,
    max_size: usize,
) -> Possibility<Vec<T>> {
    let produce = move |tc: &mut TestCase| {
        let mut result = vec![];
        loop {
            if result.len() < min_size {
                tc.det_choice(1)?;
            } else if result.len() + 1 >= max_size {
                tc.det_choice(0)?;
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

fn just<T: std::clone::Clone + 'static>(value: T) -> Possibility<T> {
    Box::new(move |_tc: &mut TestCase| { Ok(value.clone()) })
}

fn nothing<T>() -> Possibility<T> {
    Box::new(|tc: &mut TestCase| { Err(tc.reject()) })
}

fn mix_of<T: 'static>(first: Possibility<T>, second: Possibility<T>) -> Possibility<T> {
    Box::new(move |tc: &mut TestCase| {
        if tc.choice(1)? == 0 { tc.any(&first) } else { tc.any(&second) }
    })
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
    let ls: Vec<i64> = tc.any(&vecs(integers(95, 105), 9, 11)).unwrap();
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
