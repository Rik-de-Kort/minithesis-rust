/// For this we define a test as a routine returning true or false. True for passing.
use std::error::Error;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;


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
enum MinithesisError { 
	Frozen,
	StopTest
}


fn accept<T, U: Sized>(test: U) -> Option<Box<dyn Error>> {
	None
}

struct TestingState<T: ?Sized, U> {
	prefix: Vec<u8>,
	max_examples: u32,
	__test_function: Box<T>,
	valid_test_cases: u32,
	calls: u32,
	result: Option<U>,  // TODO: figure out what U should be
	test_is_trivial: bool
}

// #[derive(Clone, Copy)]
enum Status {
	Overrun,
	Invalid,
	Valid,
	Interesting(u64),
} 

struct TestCase {
	prefix: Vec<u64>,
	random: ChaCha8Rng,
	max_size: usize,
	choices: Vec<u64>,
	status: Option<Status>,
	depth: u64,
}

impl TestCase {
	fn make_choice<F: FnMut(&mut ChaCha8Rng) -> u64>(&mut self, n: u64, mut rnd_method: F) -> Result<u64, MinithesisError> {
		match &self.status {
			Some(s) => Err(MinithesisError::Frozen),
			None => { 
				if self.choices.len() >= self.max_size { 
					Err(self.mark_status(Status::Overrun))
				} else if self.choices.len() < self.prefix.len() {
					Ok(self.prefix[self.choices.len()])
				} else {
					Ok(rnd_method(&mut self.random))
				}	
			}
		}
	}

	pub fn choice(&mut self, n: u64) -> u64 {
		self.make_choice(n, |r: &mut ChaCha8Rng| r.gen_range(0, n)).unwrap()
	}

	fn mark_status(&mut self, status: Status) -> MinithesisError {
		match &self.status {
			None => {self.status = Some(status); MinithesisError::StopTest}
			Some(s) => MinithesisError::Frozen
		}
    }
}




fn main() {
	let mut tc = TestCase{
		prefix: vec![], 
		random: ChaCha8Rng::seed_from_u64(0),
		max_size: 10000,
		choices: vec![],
		status: None,
		depth: 0
	};
	println!("{}", tc.choice(15));

    println!("Hello, world!");
}


