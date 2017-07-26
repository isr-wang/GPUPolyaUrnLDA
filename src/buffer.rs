use libc::{uint32_t, c_void};
use std::mem;
use std::slice;
use std::ptr;

#[allow(dead_code)] // remove later
#[repr(C)]
pub struct Buffer {
  size: uint32_t,
  z: *const uint32_t,
  w: *const uint32_t,
  d: *const uint32_t,
  n_docs: uint32_t,
  gpu_z: *mut c_void,
  gpu_w: *mut c_void,
  gpu_d_len: *mut c_void,
  gpu_d_idx: *mut c_void,
  gpu_rng: *mut c_void,
  stream: *mut c_void,
}

#[allow(dead_code)] // remove later
impl Buffer {
  pub fn new(size: usize) -> Buffer {
    // allocate arrays
    let z = Vec::with_capacity(size).into_boxed_slice();
    let w = Vec::with_capacity(size).into_boxed_slice();
    let d = Vec::with_capacity(size).into_boxed_slice();
    // create buffer
    let b = Buffer {
      size: size as u32,
      z: z.as_ptr(),
      w: w.as_ptr(),
      d: d.as_ptr(),
      n_docs: 0,
      gpu_z: ptr::null_mut(),
      gpu_w: ptr::null_mut(),
      gpu_d_len: ptr::null_mut(),
      gpu_d_idx: ptr::null_mut(),
      gpu_rng: ptr::null_mut(),
      stream: ptr::null_mut(),
    };
    // as_ptr doesn't take ownership, we need to be sure not to deallocate any arrays
    mem::forget(z);
    mem::forget(w);
    mem::forget(d);
    // return the buffer
    b
  }
}

#[allow(unused_variables)]
impl Drop for Buffer {
  fn drop(&mut self) {
    unsafe {
      let z = Box::new(slice::from_raw_parts(self.z, self.size as usize));
      let w = Box::new(slice::from_raw_parts(self.w, self.size as usize));
      let d = Box::new(slice::from_raw_parts(self.d, self.size as usize));
      // at this point, z,w,dLen,dIdx are dropped and deallocated
    }
  }
}
