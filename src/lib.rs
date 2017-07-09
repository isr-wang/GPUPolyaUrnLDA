extern crate clap;
extern crate libc;
extern crate rand;

mod preprocess;

#[cfg(test)]
mod preprocess_test {

    #[test]
    fn this_test_will_pass() {
        let value = 10;
        assert_eq!(10, value);
    }

//    #[test]
//    fn this_test_will_fail() {
//        let value = 10;
//        assert_eq!(5, value);
//    }

    #[test]
    fn test_get_tokens() {
        preprocess::get_tokens("hi alex and kunal");
    }
}
