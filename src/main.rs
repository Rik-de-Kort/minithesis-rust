use rand::prelude::*;

mod test_harness;
use test_harness::*;

use data;

/// We are seeing if things are interesting (true or false)
/// Note that we invert the relationship: true means "interesting"
fn example_test(tc: &mut TestCase) -> bool {
    let ls = tc.any(&data::vectors(data::integers(95, 105), 9, 11));
    match ls {
        Ok(list) => list.iter().sum::<i64>() > 1000,
        Err(_) => false,
    }
}


fn main() {
    let mut ts = TestState::new(thread_rng(), Box::new(example_test), 10);
    ts.run();
    println!("Test result {:?}", ts.result);

    let mut tc = TestCase::for_choices(&ts.result.unwrap());
    println!(
        "Final list {:?}",
        tc.any(&data::vectors(data::integers(95, 105), 9, 11))
    );
}
