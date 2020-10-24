use rand::prelude::*;
use crate::test_harness::*;
use crate::data::*;

fn vecs_of_ints() -> Vectors<i64, Integers> {
    vectors(integers(0, 10000), 0, 10000)
}

fn finds_small_list(tc: &mut TestCase) -> bool {
    match tc.any(&vecs_of_ints()) {
        Ok(ls) => ls.iter().sum::<i64>() > 1000,
        Err(_) => false,
    }
}


#[test]
fn test_finds_small_list() {
    let mut ts = TestState::new(thread_rng(), Box::new(finds_small_list), 10000);
    ts.run();
    println!("Test result {:?}", ts.result);
    let result = ts.get_value_for(&vecs_of_ints());
    let expected = Some(vec![1001]);
    assert_eq!(result, expected);
}

use std::convert::TryInto;

struct BadList;
impl Possibility<Vec<i64>> for BadList {
    fn produce(&self, tc: &mut TestCase) -> Result<Vec<i64>, MTErr> {
        let n = tc.choice(10)?;
        // Todo: this code is ugly as sin
        let result: Vec<u64> = (0..n).map(|_| {tc.choice(10000)}).collect::<Result<Vec<u64>, MTErr>>()?;
        Ok(result.iter().map(|i| *i as i64).collect())
    }
}

