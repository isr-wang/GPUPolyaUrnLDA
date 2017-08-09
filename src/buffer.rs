use args::ARGS;
use libc::{uint32_t, c_void};
use std::mem;
use std::slice;
use std::ptr;

#[allow(dead_code)] // remove later
#[allow(non_snake_case)]
#[repr(C)]
pub struct Buffer {
  z: *const uint32_t,
  w: *const uint32_t,
  d: *const uint32_t,
  K_d: *const uint32_t,
  n_docs: uint32_t,
  gpu_z: *mut c_void,
  gpu_w: *mut c_void,
  gpu_d_len: *mut c_void,
  gpu_d_idx: *mut c_void,
  gpu_K_d: *mut c_void,
  gpu_rng: *mut c_void,
  stream: *mut c_void,
}

#[allow(dead_code)] // remove later
#[allow(non_snake_case)]
impl Buffer {
  pub fn new() -> Buffer {
    // allocate arrays
    let z = Vec::with_capacity(ARGS.buffer_size).into_boxed_slice();
    let w = Vec::with_capacity(ARGS.buffer_size).into_boxed_slice();
    let d = Vec::with_capacity(ARGS.buffer_max_docs).into_boxed_slice();
    let K_d = Vec::with_capacity(ARGS.buffer_max_docs).into_boxed_slice();
    // create buffer
    let b = Buffer {
      z: z.as_ptr(),
      w: w.as_ptr(),
      d: d.as_ptr(),
      K_d: K_d.as_ptr(),
      n_docs: 0,
      gpu_z: ptr::null_mut(),
      gpu_w: ptr::null_mut(),
      gpu_d_len: ptr::null_mut(),
      gpu_d_idx: ptr::null_mut(),
      gpu_K_d: ptr::null_mut(),
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
#[allow(non_snake_case)]
impl Drop for Buffer {
  fn drop(&mut self) {
    unsafe {
      let z = Box::new(slice::from_raw_parts(self.z, ARGS.buffer_size));
      let w = Box::new(slice::from_raw_parts(self.w, ARGS.buffer_size));
      let d = Box::new(slice::from_raw_parts(self.d, ARGS.buffer_max_docs));
      let K_d = Box::new(slice::from_raw_parts(self.K_d, ARGS.buffer_max_docs));
      // at this point, z,w,d,K_d are dropped and deallocated
    }
  }
}
