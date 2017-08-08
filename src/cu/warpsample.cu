#include "warpsample.cuh"
#include <thrust/system/cuda/detail/cub/block/block_scan.cuh>
//#include <thrust/scan.h>
//#include <thrust/device_ptr.h>
//#include <thrust/execution_policy.h>

namespace gplda {

__global__ void compute_d_idx(uint32_t* d_len, uint32_t* d_idx, uint32_t n_docs) {
  typedef cub::BlockScan<int32_t, GPLDA_COMPUTE_D_IDX_BLOCKDIM> BlockScan;
  __shared__ typename BlockScan::TempStorage temp;

  if(blockIdx.x == 0) {
    int32_t thread_d;
    int32_t initial_value = 0;
    int32_t total_value;
    for(int32_t offset = 0; offset < n_docs / blockDim.x + 1; ++offset) {
      int32_t i = threadIdx.x + offset * blockDim.x;
      if(i < n_docs) {
        thread_d = d_len[i];
      } else {
        thread_d = 0;
      }

      BlockScan(temp).ExclusiveScan(thread_d, thread_d, 0, cub::Sum(), total_value);

      // workaround for CUB bug: apply offset manually
      __syncthreads();
      thread_d = thread_d + initial_value;
      initial_value = total_value + initial_value;

      if(i < n_docs) {
        d_idx[i] = thread_d;
      }
    }
  }
}

__global__ void warp_sample_topics(uint32_t size, uint32_t n_docs, uint32_t *z, uint32_t *w, uint32_t *d_len, uint32_t *d_idx, float** prob, float** alias, curandStatePhilox4_32_10_t* rng) {
//  __shared__ uint32_t m_topics[128];
//  __shared__ uint32_t m_words[128];

  // load current row of Phi into shared memory
  // load Alias table into shared memory (worth it? we may not access it at all)
  // compute
}

}
