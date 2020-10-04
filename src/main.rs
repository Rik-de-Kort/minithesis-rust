/// For this we define a test as a routine returning true or false. True for passing.
use std::error::Error;
use rand_chacha::ChaCha8Rng;

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

enum Status {
	Overrun,
	Invalid,
	Valid,
	Interesting(u64),
} 

struct TestCase {
	prefix: Vec<u8>,
	random: ChaCha8Rng,
	max_size: u32,
	choices: Vec<u8>,
	status: Option<Status>,
	depth: u32,
}

impl TestCase {
	fn choice(self, n: u32) -> u32 {
		n
	}
}



fn f(x: f64) -> f64 { x * 2.0 }


fn main() {
	let g = |x: f64| { x * 2.0 };
	let ts = TestingState{
		prefix: vec![],
		max_examples: 10000,
		__test_function: Box::new(g),
		valid_test_cases: 0,  // Todo: these four are all set on init
		calls: 0,
		result: Some(true),
		test_is_trivial: false
	};
    println!("Hello, world!");
}
