use rand::prelude::*;

#[derive(Debug)]
pub enum MTErr {
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

pub struct TestCase {
    prefix: Vec<u64>,
    random: ThreadRng,
    max_size: usize,
    choices: Vec<u64>,
    status: Option<MTStatus>,
    // print_results: bool,
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
        p.produce(self).map(|val| {
            self.depth += 1;
            val
        })
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

    /// Return an integer in the range [0, n]
    fn choice(&mut self, n: u64) -> Result<u64, MTErr> {
        if self.choices.len() < self.prefix.len() {
            self.det_choice(self.prefix[self.choices.len()])
        } else {
            let result = self.random.gen_range(0, n + 1);
            self.det_choice(result)
        }
    }
}

mod data {
    /// Represents some range of values that might be used in a test, that can be requested from a
    /// TestCase.
    use crate::*;
    use std::convert::TryInto;
    use std::marker::{PhantomData, Sized};
    use std::clone::Clone;

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

    impl Possibility<i64> for Integers {
        fn produce(&self, tc: &mut TestCase) -> Result<i64, MTErr> {
            let offset: i64 = tc.choice(self.range)?.try_into().unwrap();
            Ok(self.minimum + offset)
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

    struct Nothing {}
    impl<T> Possibility<T> for Nothing {
        fn produce(&self, tc: &mut TestCase) -> Result<T, MTErr> {
            Err(tc.reject())
        }
    }

    struct MixOf<T, P: Possibility<T>> {
        first: P,
        second: P,
        phantom_t: PhantomData<T>,
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
}

/// We are seeing if things are interesting (true or false)
type InterestingTest<T> = Box<dyn Fn(&mut T) -> bool>;

struct TestState {
    random: ThreadRng,
    max_examples: usize,
    is_interesting: InterestingTest<TestCase>,
    valid_test_cases: usize,
    calls: usize,
    result: Option<Vec<u64>>,
    best_scoring: Option<Vec<u64>>, // Todo: check type
    test_is_trivial: bool,
}

const BUFFER_SIZE: usize = 8 * 1024;

impl TestState {
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
                self.test_is_trivial = test_case.choices.is_empty();
            }
            Some(MTStatus::Valid) => {
                self.test_is_trivial = test_case.choices.is_empty();
                self.valid_test_cases += 1;
            }
            Some(MTStatus::Interesting) => {
                self.test_is_trivial = test_case.choices.is_empty();
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
            self.test_function(TestCase::new(vec![], self.random, BUFFER_SIZE));
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
        .any(&data::Vectors::new(data::Integers::new(95, 105), 9, 11))
        .unwrap();
    println!("running with list {:?}", ls);
    ls.iter().sum::<i64>() > 1000
}

fn main() {
    let mut ts = TestState {
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
