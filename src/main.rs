mod test_harness;
mod data;

use rand::prelude::*;
use crate::test_harness::*;

/// We are seeing if things are interesting (true or false)
/// Note that we invert the relationship: true means "interesting"
fn example_test(tc: &mut TestCase) -> bool {
    let ls = tc.any(&data::vectors(data::integers(0, 10000), 0, 10000));
    match ls {
        Ok(list) => list.iter().sum::<i64>() > 1000,
        Err(_) => false,
    }
}


fn main() {
    let mut ts = TestState::new(thread_rng(), Box::new(example_test), 10000);
    ts.run();
    println!("Test result {:?}", ts.result);

    let mut tc = TestCase::for_choices(&ts.result.unwrap());
    println!(
        "Final list {:?}",
        tc.any(&data::vectors(data::integers(0, 10000), 0, 10000))
    );
}

#[cfg(test)]
mod test;
