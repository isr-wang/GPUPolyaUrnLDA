mod args;
mod preprocess;
mod train;
mod output;
mod buffer;

extern crate libc;

fn main() {
  println!("Hello, world!");
  args::parse();
  preprocess::preprocess();
  train::train();
  output::output();
  let b = buffer::Buffer::new(10);
}
