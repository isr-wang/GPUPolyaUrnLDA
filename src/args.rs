pub struct Args {
  pub alpha: f32,
  pub beta: f32,
  pub K: u32,
  pub nMC: u32,
  pub seed: u64,
  pub bufferSize: i32,
  pub input: String,
  pub output: String,
  pub zTempFile: String,
  pub wTempFile: String,
  pub dTempFile: String,
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
