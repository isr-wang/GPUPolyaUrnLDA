#include "polyaurn.cuh"
#include "assert.h"

namespace gplda {

__device__ __forceinline__ float draw_poisson(float u, float beta, uint32_t n,
    float** prob, float** alias, uint32_t max_lambda, uint32_t max_value) {
  // MUST be defined in this file to compile on all platforms

  // if below cutoff, draw using Alias table
  if(n < max_lambda) {
    // determine the slot and update random number
    float mv = (float) max_value;
    uint32_t slot = (uint32_t) (u * mv);
    u = fmodf(u, __frcp_rz(mv)) * mv;

    // load table elements from global memory
    float thread_prob = prob[n][slot];
    float thread_alias = alias[n][slot];

    // return the resulting draw
    if(u < thread_prob) {
      return (float) slot;
    } else {
      return thread_alias;
    }
  }

  // if didn't return, draw using Gaussian approximation
  if(u == 1.0f || u == 0.0f) {
    u = 0.5f; // prevent overflow edge cases
  }
  float mu = beta + ((float) n);
  return normcdfinvf(u) * mu + mu;
}

__device__ __forceinline__ float block_reduce_sum(float* block_sum, float thread_sum) {
  // first, perform a warp reduce
  for(int offset = warpSize/2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down(thread_sum, offset);
  }

  // then, add result to shared memory
  if(threadIdx.x % warpSize == 0) {
    atomicAdd(block_sum, thread_sum);
  }

  // ensure all threads finish writing
  __syncthreads();

  // return new value to all threads
  return block_sum[0];
}

__global__ void polya_urn_sample(float* Phi, uint32_t* n, float beta, uint32_t V,
    float** prob, float** alias, uint32_t max_lambda, uint32_t max_value) {
  // initialize variables
  float thread_sum = 0.0f;
  __shared__ float block_sum[1];
  if(threadIdx.x == 0) {
    block_sum[0] = 0.0f;
  }

  // loop over array and draw samples
  for(int offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int col = threadIdx.x + offset * blockDim.x;
    int array_idx = col + V * blockIdx.x;
    if(col < V) {
      float pois = draw_poisson(Phi[array_idx], beta, n[array_idx], prob, alias, max_lambda, max_value);
      Phi[array_idx] = pois;
      thread_sum += pois;
    }
  }

  // add up thread sums, synchronize, and broadcast
  thread_sum = block_reduce_sum(block_sum, thread_sum);

  // normalize draws
  for(int offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int col = threadIdx.x + offset * blockDim.x;
    int array_idx = col + V * blockIdx.x;
    if(col < V) {
      Phi[array_idx] /= thread_sum;
    }
  }
}

__global__ void polya_urn_colsums(float* Phi, float* sigma_a, float** prob, uint32_t K) {  // initilize variables
  // initialize variables
  float thread_sum = 0.0f;
  __shared__ float block_sum[1];
  if(threadIdx.x == 0) {
    block_sum[0] = 0.0f;
  }

  // loop over array and compute column sums
  for(int offset = 0; offset < K / blockDim.x + 1; ++offset) {
    int row = threadIdx.x + offset * blockDim.x;
    int array_idx = row + K * blockIdx.x;
    if(row < K) {
      thread_sum += Phi[array_idx];
    }
  }

  // add up thread sums, synchronize, and broadcast
  thread_sum = block_reduce_sum(block_sum, thread_sum);

  // set sigma_a
  if(threadIdx.x == 0) {
    sigma_a[blockIdx.x] = thread_sum;
  }

  // compute and set Alias table probabilities
  for(int offset = 0; offset < K / blockDim.x + 1; ++offset) {
    int row = threadIdx.x + offset * blockDim.x;
    int array_idx = row + K * blockIdx.x;
    if(row < K) {
      prob[blockIdx.x][row] = Phi[array_idx] / thread_sum;
    }
  }
}

}
