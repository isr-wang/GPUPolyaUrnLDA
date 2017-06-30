pub struct Args {
  alpha: f32,
  beta: f32,
  K: u32,
  nMC: u32,
  seed: u64,
  bufferSize: i32,
  input: String,
  output: String,
  zTempFile: String,
  wTempFile: String,
  dTempFile: String,
}

pub fn parse() -> Args {
  Args {
    alpha: 0.1,
    beta: 0.1,
    K: 10,
    nMC: 100,
    seed: 0,
    bufferSize: 1024,
    input: "data/small.txt".to_string(),
    output: "output/small.txt".to_string(),
    zTempFile: "temp/z.bin".to_string(),
    wTempFile: "temp/w.bin".to_string(),
    dTempFile: "temp/d.bin".to_string(),
  }
}
