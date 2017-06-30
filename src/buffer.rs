use libc::{uint32_t, size_t};
use std::mem;
use std::slice;

#[repr(C)]
pub struct Buffer {
  size: size_t,
  z: *const uint32_t,
  w: *const uint32_t,
  dLen: *const uint32_t,
  dIdx: *const uint32_t,
  nDocs: uint32_t,
}

impl Buffer {
  pub fn new(size: usize) -> Buffer {
    // allocate arrays
    let mut z = Vec::with_capacity(size).into_boxed_slice();
    let mut w = Vec::with_capacity(size).into_boxed_slice();
    let mut dLen = Vec::with_capacity(size).into_boxed_slice();
    let mut dIdx = Vec::with_capacity(size).into_boxed_slice();
    // create buffer
    let b = Buffer {
      size: size,
      z: z.as_mut_ptr(),
      w: w.as_mut_ptr(),
      dLen: dLen.as_mut_ptr(),
      dIdx: dIdx.as_mut_ptr(),
      nDocs: 0,
    };
    // as_mut_ptr doesn't take ownership, we need to be sure not to deallocate any arrays
    mem::forget(z);
    mem::forget(w);
    mem::forget(dLen);
    mem::forget(dIdx);
    // return the buffer
    b
  }
}

impl Drop for Buffer {
  fn drop(&mut self) {
    unsafe {
      let z = slice::from_raw_parts(self.z, self.size);
      let w = slice::from_raw_parts(self.w, self.size);
      let dLen = slice::from_raw_parts(self.dLen, self.size);
      let dIdx = slice::from_raw_parts(self.dIdx, self.size);
      // at this point, z,w,dLen,dIdx are dropped and deallocated
    }
  }
}
