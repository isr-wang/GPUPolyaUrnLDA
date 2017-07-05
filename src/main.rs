extern crate clap;
extern crate libc;
extern crate rand;
#[macro_use]
extern crate lazy_static;

mod args;
mod preprocess;
mod train;
mod output;
mod buffer;

fn main() {
  println!("Hello, world!");
  preprocess::preprocess();
  train::train();
  output::output();
}
