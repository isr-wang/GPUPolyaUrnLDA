extern crate clap;
extern crate libc;
extern crate rand;

mod args;
mod preprocess;
mod train;
mod output;
mod buffer;

fn main() {
  println!("Hello, world!");
  let args = args::parse();
  preprocess::preprocess(&args);
  train::train(&args);
  output::output(&args);
}
