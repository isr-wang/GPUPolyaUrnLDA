#include "test_warpsample.cuh"
#include "../warpsample.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;

namespace gplda_test {

void test_compute_d_idx() {
  uint32_t size = 4*GPLDA_COMPUTE_D_IDX_BLOCKDIM;
  uint32_t d_len[4*GPLDA_COMPUTE_D_IDX_BLOCKDIM];
  uint32_t d_idx[4*GPLDA_COMPUTE_D_IDX_BLOCKDIM];
  uint32_t n_docs = 2*GPLDA_COMPUTE_D_IDX_BLOCKDIM + 15;

  for(int32_t i = 0; i < size; ++i) {
    d_len[i] = i+1;
  }

  uint32_t* gpu_d_len;
  uint32_t* gpu_d_idx;
  cudaMalloc(&gpu_d_len, size*sizeof(uint32_t)) >> GPLDA_CHECK;
  cudaMalloc(&gpu_d_idx, size*sizeof(uint32_t)) >> GPLDA_CHECK;

  cudaMemcpy(gpu_d_len, d_len, size*sizeof(uint32_t), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::compute_d_idx<<<1,GPLDA_COMPUTE_D_IDX_BLOCKDIM>>>(gpu_d_len, gpu_d_idx, n_docs);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(d_idx, gpu_d_idx, size*sizeof(uint32_t), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(d_idx[0] == 0);
  uint32_t j = d_len[0];
  for(int32_t i = 1; i < n_docs; ++i) {
     assert(d_idx[i] == j);
     j = j + d_len[i];
   }

  cudaFree(gpu_d_len);
  cudaFree(gpu_d_idx);
}

void test_warp_sample_topics() {

}

}
