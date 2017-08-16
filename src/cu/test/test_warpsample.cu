#include "test_warpsample.cuh"
#include "../warpsample.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;
using gplda::f32;
using gplda::i32;
using gplda::u32;
using gplda::u64;

namespace gplda_test {

void test_compute_d_idx() {
  u32 size = 4*GPLDA_COMPUTE_D_IDX_BLOCKDIM;
  u32 d_len[4*GPLDA_COMPUTE_D_IDX_BLOCKDIM];
  u32 d_idx[4*GPLDA_COMPUTE_D_IDX_BLOCKDIM];
  u32 n_docs = 2*GPLDA_COMPUTE_D_IDX_BLOCKDIM + 15;

  for(i32 i = 0; i < size; ++i) {
    d_len[i] = i+1;
  }

  u32* gpu_d_len;
  u32* gpu_d_idx;
  cudaMalloc(&gpu_d_len, size*sizeof(u32)) >> GPLDA_CHECK;
  cudaMalloc(&gpu_d_idx, size*sizeof(u32)) >> GPLDA_CHECK;

  cudaMemcpy(gpu_d_len, d_len, size*sizeof(u32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::compute_d_idx<<<1,GPLDA_COMPUTE_D_IDX_BLOCKDIM>>>(gpu_d_len, gpu_d_idx, n_docs);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(d_idx, gpu_d_idx, size*sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(d_idx[0] == 0);
  u32 j = d_len[0];
  for(i32 i = 1; i < n_docs; ++i) {
     assert(d_idx[i] == j);
     j = j + d_len[i];
   }

  cudaFree(gpu_d_len);
  cudaFree(gpu_d_idx);
}

void test_warp_sample_topics() {

}

}
