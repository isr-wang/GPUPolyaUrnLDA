use args::Args;
use std::fs::{File, remove_file};
use std::io::prelude::{BufRead, Write};
use std::io::{BufReader, BufWriter};
use rand::{sample, thread_rng};
use rand::distributions::{Range, IndependentSample};
use std::slice;
use std::mem;
use std::collections::HashMap;


fn get_tokens(line: String) -> Vec<String> {
    line.split("\t").skip(2).next().unwrap().split_whitespace().map(|s| s.to_string()).collect()
}

fn count_tokens(args: &Args) -> HashMap<String, u32> {
    let mut count: HashMap<String, u32> = HashMap::new();
    let input = BufReader::new(File::open(&args.input).unwrap());
    for line in input.lines() {
        get_tokens(line.unwrap()).iter().fold((),|_,token| *count.entry(token.clone()).or_insert(0) += 1);
    }
    let mut count_vec: Vec<(String, u32)> = count.iter().map(|t| (t.0.clone(),t.1.clone())).collect();
    count_vec.sort_by(|t1,t2| t2.1.cmp(&t1.1));
    let mut c = BufWriter::new(File::create(&args.cTempFile).unwrap());
    let mut token_ids: HashMap<String, u32> = HashMap::new();
    for (idx,&(ref token,count)) in count_vec.iter().enumerate() {
        write!(&mut c, "{}\t{}\t{}\n", token, idx, count).unwrap();
        token_ids.insert(token.clone(), idx as u32);
    }
    token_ids
}

fn as_bytes<'a>(v: &'a Vec<u32>) -> &'a [u8] {
    let s = v.as_slice();
    unsafe {
        slice::from_raw_parts(s.as_ptr() as *const u8,
                              s.len() * mem::size_of::<u32>() / mem::size_of::<u8>())
    }
}

pub fn preprocess(args: &Args) {
    println!("preprocessing");
    remove_file(&args.zTempFile).unwrap_or_else(|_| ());
    remove_file(&args.wTempFile).unwrap_or_else(|_| ());
    remove_file(&args.dTempFile).unwrap_or_else(|_| ());

    let token_ids = count_tokens(&args);

    let input = BufReader::new(File::open(&args.input).unwrap());
    let mut z = BufWriter::new(File::create(&args.zTempFile).unwrap());
    let mut w = BufWriter::new(File::create(&args.wTempFile).unwrap());
    let mut d = BufWriter::new(File::create(&args.dTempFile).unwrap());

    let mut rand = thread_rng();
    let unif = Range::new(0, args.K);

    for line in input.lines() {
        let tokens = get_tokens(line.unwrap());
        z.write_all(as_bytes(&(0..tokens.len()).map(|_| unif.ind_sample(&mut rand)).collect())).unwrap();
        w.write_all(as_bytes(&tokens.iter().map(|t| *token_ids.get(t).unwrap()).collect())).unwrap();
        d.write_all(as_bytes(&vec![tokens.len() as u32])).unwrap();
    }

    z.flush().unwrap(); w.flush().unwrap(); d.flush().unwrap();
}
