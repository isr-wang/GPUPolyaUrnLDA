#include "warpsample.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace gplda {

void compute_d_idx(cudaStream_t& stream, uint32_t* d_len, uint32_t* d_idx, uint32_t n_docs) {
  // this is EXACTLY a cumulative sum, so use Thrust library
  thrust::device_ptr<uint32_t> thrust_d_len_ptr(d_len);
  thrust::device_ptr<uint32_t> thrust_d_idx_ptr(d_idx);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust_d_len_ptr, thrust_d_len_ptr + n_docs, thrust_d_idx_ptr);
}

__global__ void warp_sample_topics(uint32_t size, uint32_t n_docs, uint32_t *z, uint32_t *w, uint32_t *d_len, uint32_t *d_idx, curandStatePhilox4_32_10_t* rng) {
  // load current row of Phi into shared memory
  // load Alias table into shared memory (worth it? we may not access it at all)
  // compute
}

}
