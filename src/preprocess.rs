use args::Args;
use serialize::as_bytes;
use std::fs::{File, remove_file};
use std::io::prelude::{BufRead, Write};
use std::io::{BufReader, BufWriter};
use rand::{sample, thread_rng};
use rand::distributions::{Range, IndependentSample};

fn get_token_id(token: &String) -> u32 {
    unimplemented!();
}

fn get_tokens(line: String) -> Vec<String> {
    unimplemented!();
}

fn count() {
    unimplemented!();
}

pub fn preprocess(args: &Args) {
    println!("preprocessing");
    remove_file(&args.zTempFile).unwrap_or_else(|_| ());
    remove_file(&args.wTempFile).unwrap_or_else(|_| ());
    remove_file(&args.dTempFile).unwrap_or_else(|_| ());

    count();

    let input = BufReader::new(File::open(&args.input).unwrap());
    let mut z = BufWriter::new(File::create(&args.zTempFile).unwrap());
    let mut w = BufWriter::new(File::create(&args.wTempFile).unwrap());
    let mut d = BufWriter::new(File::create(&args.dTempFile).unwrap());

    let mut rand = thread_rng();
    let unif = Range::new(0, args.K);

    for line in input.lines() {
        let tokens = get_tokens(line.unwrap());
        z.write_all(as_bytes(&(0..tokens.len()).map(|_| unif.ind_sample(&mut rand)).collect())).unwrap();
        w.write_all(as_bytes(&tokens.iter().map(|t| get_token_id(t)).collect())).unwrap();
        d.write_all(as_bytes(&vec![tokens.len() as u32])).unwrap();
    }

    z.flush().unwrap(); w.flush().unwrap(); d.flush().unwrap();
}
