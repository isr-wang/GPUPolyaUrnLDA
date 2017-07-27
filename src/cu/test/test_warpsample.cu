#include "test_warpsample.cuh"
#include "../warpsample.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;

namespace gplda_test {

void test_compute_d_idx() {
  uint32_t size = 15;
  uint32_t d_len[15] = {3,2,4,2,0, 0,0,0,0,0, 0,0,0,0,0};
  uint32_t d_idx[15] = {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
  uint32_t n_docs = 4;

  uint32_t* gpu_d_len;
  uint32_t* gpu_d_idx;
  cudaMalloc(&gpu_d_len, size*sizeof(uint32_t)) >> GPLDA_CHECK;
  cudaMalloc(&gpu_d_idx, size*sizeof(uint32_t)) >> GPLDA_CHECK;

  cudaMemcpy(gpu_d_len, d_len, size*sizeof(uint32_t), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  cudaStream_t* stream = new cudaStream_t;
  cudaStreamCreate(stream) >> GPLDA_CHECK;

  gplda::compute_d_idx(*stream, gpu_d_len, gpu_d_idx, n_docs);
  cudaStreamSynchronize(*stream);

  cudaMemcpy(d_idx, gpu_d_idx, size*sizeof(uint32_t), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(d_idx[0] == 0);
  assert(d_idx[1] == 3);
  assert(d_idx[2] == 5);
  assert(d_idx[3] == 9);

  cudaStreamDestroy(*stream);
  delete stream;

  cudaFree(gpu_d_len);
  cudaFree(gpu_d_idx);
}

void test_warp_sample_topics() {

}

}
