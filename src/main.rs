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
enum MTErr { 
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
	fn make_choice<F: FnMut(&mut ChaCha8Rng) -> u64>(&mut self, n: u64, mut rnd_method: F) -> Result<u64, MTErr> {
		match &self.status {
			Some(s) => Err(MTErr::Frozen),
			None => { 
				if self.choices.len() >= self.max_size { 
					Err(self.mark_status(Status::Overrun))
				} else if self.choices.len() < self.prefix.len() {
                    let choice = self.prefix[self.choices.len()];
                    self.choices.push(choice);
					Ok(self.prefix[self.choices.len()])
				} else {
                    let choice = rnd_method(&mut self.random);
                    if choice > n { 
                        Err(self.mark_status(Status::Invalid))
                    } else {
                        self.choices.push(choice);
                        Ok(choice)
                    }
				}	
			}
		}
	}

	pub fn choice(&mut self, n: u64) -> u64 {
		self.make_choice(n, |r: &mut ChaCha8Rng| r.gen_range(0, n)).unwrap()
	}

    pub fn forced_choice(&mut self, n: u64) -> Result<u64, MTErr> {
        match &self.status {
            Some(s) => Err(MTErr::Frozen),
            None => {
                if self.choices.len() >= self.max_size {
                    Err(self.mark_status(Status::Overrun))
                } else {
                    self.choices.push(n);
                    Ok(n)
                }
            }
        }
    }

    pub fn weighted(&mut self, p: f64) -> bool {
        if p <= 0.0 {
            self.forced_choice(0).unwrap();
            false
        } else if p >= 1.0 {
            self.forced_choice(1).unwrap();
            true
        } else {
            let result = self.make_choice(1, |r: &mut ChaCha8Rng| r.gen_range(0, 1)).unwrap();
            match result {
                0 => false,
                _ => true
            }
        }
    }

    pub fn reject(&mut self) -> MTErr {
        self.mark_status(Status::Invalid)
    }



	fn mark_status(&mut self, status: Status) -> MTErr {
		match &self.status {
			None => {self.status = Some(status); MTErr::StopTest}
			Some(s) => MTErr::Frozen
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
    println!("{}", tc.forced_choice(110).unwrap());

    println!("Hello, world!");
}


