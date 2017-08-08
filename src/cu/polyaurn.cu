#include "polyaurn.cuh"
#include "assert.h"
#include "error.cuh"
#include <thrust/system/cuda/detail/cub/block/block_scan.cuh> // workaround for CUB missing include
#include <thrust/system/cuda/detail/cub/block/block_reduce.cuh>

namespace gplda {

__device__ __forceinline__ float draw_poisson(float u, float beta, uint32_t n,
    float** prob, uint32_t** alias, uint32_t max_lambda, uint32_t max_value) {
  // MUST be defined in this file to compile on all platforms

  // if below cutoff, draw using Alias table
  if(n < max_lambda) {
    // determine the slot and update random number
    float mv = (float) max_value;
    uint32_t slot = (uint32_t) (u * mv);
    u = fmodf(u, __frcp_rz(mv)) * mv;

    // load table elements from global memory
    float thread_prob = prob[n][slot];
    uint32_t thread_alias = alias[n][slot];

    // return the resulting draw
    if(u < thread_prob) {
      return (float) slot;
    } else {
      return (float) thread_alias;
    }
  }

  // if didn't return, draw using Gaussian approximation
  if(u == 1.0f || u == 0.0f) {
    u = 0.5f; // prevent overflow edge cases
  }
  float mu = beta + ((float) n);
  return normcdfinvf(u) * sqrtf(mu) + mu;
}


__global__ void polya_urn_init(uint32_t* n, uint32_t* C, float beta, uint32_t V,
    float** prob, uint32_t** alias, uint32_t max_lambda, uint32_t max_value,
    curandStatePhilox4_32_10_t* rng) {
  // initialize variables
  curandStatePhilox4_32_10_t thread_rng = rng[0];
  skipahead((unsigned long long int) blockIdx.x*blockDim.x + threadIdx.x, &thread_rng);

  // loop over array and draw samples
  for(int32_t offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int32_t col = threadIdx.x + offset * blockDim.x;
    int32_t array_idx = col + V * blockIdx.x;
    if(col < V) {
      // draw n_k ~ Pois(C/K + beta)
      float u = curand_uniform(&thread_rng);
      float pois = draw_poisson(u, beta, C[col] / gridDim.x/*=K*/, prob, alias, max_lambda, max_value);
      n[array_idx] = (uint32_t) pois;
    }
  }
}







__global__ void polya_urn_sample(float* Phi, uint32_t* n, float beta, uint32_t V,
    float** prob, uint32_t** alias, uint32_t max_lambda, uint32_t max_value,
    curandStatePhilox4_32_10_t* rng) {
  // initialize variables
  float thread_sum = 0.0f;
  __shared__ float block_sum[1];
  curandStatePhilox4_32_10_t thread_rng = rng[0];
  skipahead((unsigned long long int) blockIdx.x*blockDim.x + threadIdx.x, &thread_rng);
  typedef cub::BlockReduce<float, GPLDA_POLYA_URN_SAMPLE_BLOCKDIM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;

  // loop over array and draw samples
  for(int32_t offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int32_t col = threadIdx.x + offset * blockDim.x;
    int32_t array_idx = col + V * blockIdx.x;
    if(col < V) {
      float u = curand_uniform(&thread_rng);
      float pois = draw_poisson(u, beta, n[array_idx], prob, alias, max_lambda, max_value);
      Phi[array_idx] = pois;
      thread_sum += pois;
    }
  }

  // add up thread sums, synchronize, and broadcast
  thread_sum = BlockReduce(temp).Reduce(thread_sum, cub::Sum());
  if(threadIdx.x == 0) {
    block_sum[0] = thread_sum;
  }
  __syncthreads();
  thread_sum = block_sum[0];

  // normalize draws
  for(int32_t offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int32_t col = threadIdx.x + offset * blockDim.x;
    int32_t array_idx = col + V * blockIdx.x;
    if(col < V) {
      Phi[array_idx] /= thread_sum;
    }
  }
}



void polya_urn_transpose(cudaStream_t* stream, float* Phi, float* Phi_temp, uint32_t K, uint32_t V, cublasHandle_t* handle, float* d_zero, float* d_one) {
  cudaMemcpyAsync(Phi_temp, Phi, V * K * sizeof(float), cudaMemcpyDeviceToDevice, *stream) >> GPLDA_CHECK;
  cublasSetStream(*handle, *stream) >> GPLDA_CHECK; //
  cublasSgeam(*handle, CUBLAS_OP_T, CUBLAS_OP_N, K, V, d_one, Phi_temp, V, d_zero, Phi, K, Phi, K) >> GPLDA_CHECK;
}

__global__ void polya_urn_reset(uint32_t* n, uint32_t V) {
  for(int32_t offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int32_t col = threadIdx.x + offset * blockDim.x;
    int32_t array_idx = col + V * blockIdx.x;
    if(col < V) {
      n[array_idx] = 0;
    }
  }
}


__global__ void polya_urn_colsums(float* Phi, float* sigma_a, float alpha, float** prob, uint32_t K) {  // initilize variables
  // initialize variables
  float thread_sum = 0.0f;
  __shared__ float block_sum[1];
  typedef cub::BlockReduce<float, GPLDA_POLYA_URN_COLSUMS_BLOCKDIM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;

  // loop over array and compute column sums
  for(int32_t offset = 0; offset < K / blockDim.x + 1; ++offset) {
    int32_t row = threadIdx.x + offset * blockDim.x;
    int32_t array_idx = row + K * blockIdx.x;
    if(row < K) {
      thread_sum += Phi[array_idx];
    }
  }

  // add up thread sums, synchronize, and broadcast
  thread_sum = BlockReduce(temp).Reduce(thread_sum, cub::Sum());
  if(threadIdx.x == 0) {
    block_sum[0] = thread_sum;
  }
  __syncthreads();
  thread_sum = block_sum[0];

  // set sigma_a
  if(threadIdx.x == 0) {
    sigma_a[blockIdx.x] = alpha * thread_sum;
  }

  // compute and set alias table probabilities
  for(int32_t offset = 0; offset < K / blockDim.x + 1; ++offset) {
    int32_t row = threadIdx.x + offset * blockDim.x;
    int32_t array_idx = row + K * blockIdx.x;
    if(row < K) {
      prob[blockIdx.x][row] = Phi[array_idx] / thread_sum;
    }
  }
}

}
