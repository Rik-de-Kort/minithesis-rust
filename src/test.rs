use rand::prelude::*;
use crate::test_harness::*;
use crate::data::*;

#[test]
fn test_finds_small_list() {
    fn sum_greater_1000(tc: &mut TestCase) -> bool {
        match tc.any(&vectors(integers(0, 10000), 0, 1000)) {
            Ok(ls) => ls.iter().sum::<i64>() > 1000,
            Err(_) => false,
        }
    }
    let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
    ts.run();
    println!("Test result {:?}", ts.result);
    let result = ts.get_value_for(&vectors(integers(0, 10000), 0, 1000));
    let expected = Some(vec![1001]);
    assert_eq!(result, expected);
}


#[test]
fn test_finds_small_list_even_with_bad_lists() {
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

    fn gens_bad_lists(tc: &mut TestCase) -> bool {
        match tc.any(&BadList) {
            Ok(ls) => ls.iter().sum::<i64>() > 1000,
            Err(_) => false,
        }
    }

    let mut ts = TestState::new(thread_rng(), Box::new(gens_bad_lists), 10000);
    ts.run();
    let result = ts.get_value_for(&BadList);
    let expected = Some(vec![1001]);
    assert_eq!(result, expected, "Test result {:?}", ts.result);
}


#[test]
fn test_reduces_additive_pairs() {
    fn sum_greater_1000(tc: &mut TestCase) -> bool {
        if let (Ok(n), Ok(m)) =  (tc.choice(1000), tc.choice(1000)) {
            m + n > 1000
        } else {
            false
        }
    }

    let mut ts = TestState::new(thread_rng(), Box::new(sum_greater_1000), 10000);
    ts.run();
    assert_eq!(ts.result, Some(vec![0, 1001]));
}

#[test]
fn test_test_cases_satisfy_preconditions(){
    fn satisfies_or_err(tc: &mut TestCase) -> bool {
        if let Ok(n) = tc.choice(10) {
            match tc.assume(n != 0) {
                None => false,
                Some(MTErr::StopTest) => false,
                _ => true
            }
        } else {
            false
        }
    }
    
    let mut ts = TestState::new(thread_rng(), Box::new(satisfies_or_err), 10000);
    ts.run();
    assert_eq!(ts.result, None);
}

