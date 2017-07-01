pub fn as_bytes<'a>(v: &'a Vec<u32>) -> &'a [u8] {
    let s = v.as_slice();
    let p = s.as_ptr();
    let l = s.len();
    unimplemented!();
}
