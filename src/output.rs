use args::Args;
use std::fs::{File, remove_file};
use std::io::prelude::{BufRead, Write};
use std::io::{Read, BufReader, BufWriter};
use std::collections::HashMap;
use std::slice;
use std::mem;

fn get_token_names(args: &Args) -> HashMap<u32, String> {
    let c = BufReader::new(File::open(&args.c_temp_file).unwrap());
    let mut token_ids: HashMap<u32, String> = HashMap::new();
    for line in c.lines() {
        let split_line: Vec<String> = line.unwrap().split("\t").take(2).map(|s| s.to_string()).collect();
        let token = &split_line[0];
        let token_id = &split_line[1].parse::<u32>().unwrap();
        token_ids.insert(*token_id,token.clone());
    }
    token_ids
}

fn safe_divide(a: usize, b: usize) -> usize {
    // check for integer division roundoff error at runtime
    a / b
}

fn as_u32<'a>(s: &'a [u8]) -> &'a [u32] {
    unsafe {
        slice::from_raw_parts(s.as_ptr() as *const u32,
                              safe_divide(s.len() * mem::size_of::<u8>(), mem::size_of::<u32>()))
    }
}

pub fn output(args: &Args) {
    remove_file(&args.output).unwrap_or_else(|_| ());
    let token_names = get_token_names(args);

    let mut z = BufReader::new(File::open(&args.z_temp_file).unwrap());
    let mut w = BufReader::new(File::open(&args.w_temp_file).unwrap());
    let mut d = BufReader::new(File::open(&args.d_temp_file).unwrap());
    let mut output = BufWriter::new(File::create(&args.output).unwrap());

    let mut z_bytes = z.bytes();
    let mut w_bytes = w.bytes();

    loop {
        let d_buffer_num_used_bytes = { // ensure d_buffer goes out of scope before consume
            let d_buffer = as_u32(d.fill_buf().unwrap());
            let mut d_num_read = 0;
            for d_size in d_buffer {
                let w_z_num_bytes = *d_size as usize * mem::size_of::<u32>();
                let token_id_bytes = w_bytes.by_ref().take(w_z_num_bytes).map(|b| b.unwrap()).collect::<Vec<u8>>();
                let token_ids = as_u32(token_id_bytes.as_slice());
                let topic_bytes = z_bytes.by_ref().take(w_z_num_bytes).map(|b| b.unwrap()).collect::<Vec<u8>>();
                let topics = as_u32(topic_bytes.as_slice());
                for (token_id, topic) in token_ids.iter().zip(topics.iter()) {
                    write!(&mut output, "{}:{}\t", token_names.get(token_id).unwrap(), topic).unwrap();
                }
                write!(&mut output, "\n").unwrap();
                d_num_read += 1;
            }
            safe_divide(d_num_read * mem::size_of::<u8>(), mem::size_of::<u32>()) // return number of bytes read
        };
        if d_buffer_num_used_bytes == 0 { break };
        d.consume(d_buffer_num_used_bytes);
    }

}
