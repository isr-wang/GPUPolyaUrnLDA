#include "assert.h"
#include "error.cuh"
#include "poisson.cuh"
#include "train.cuh"

namespace gplda {


__device__ __forceinline__ unsigned int lane_id_bits(int thread_idx) {
  return ((unsigned int) 1) << (thread_idx % warpSize);
}

__device__ __forceinline__ unsigned int lane_offset(unsigned int lane_bits, int thread_idx) {
  return __popc((~(((unsigned int) 4294967295) << (thread_idx % warpSize))) & lane_bits);
}

__global__ void build_poisson(float** prob, float** alias, float beta, int table_size) {
//  assert(blockDim.x == 32); // for simplicity, Poisson Alias tables are built on the warp level, so exit if misconfigured
  int lambda = blockIdx.x; // each block builds one table
  float L = lambda + beta;
  // populate PMF
  for(int offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int i = threadIdx.x + offset * blockDim.x;
    float x = i;
    if(i < table_size) {
      prob[lambda][i] = expf(x*logf(L) - L - lgammaf(x + 1));
    }
  }
  __syncthreads();
  // build array of large probabilities
  /*extern*/ __shared__ int large[200];
  __shared__ int num_large[1];
  if(threadIdx.x == 0) {
    num_large[0] = 0;
  }
  __syncthreads();
  float cutoff = 1.0/((float) table_size);
  // loop over PMF
  for(int offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int i = threadIdx.x + offset * blockDim.x;
    if(i < table_size) {
      float thread_prob = prob[lambda][i];
      // determine which threads have large probabilities
      unsigned int warp_large_bits = __ballot(thread_prob >= cutoff);
      // determine how many large probabilities are in the warp's view
      int warp_num_large = __popc(warp_large_bits);
      // increment the array's size, only once per warp, then broadcast to all lanes in the warp
      int warp_large_start;
      if(threadIdx.x % warpSize == 0) {
        warp_large_start = atomicAdd(num_large, warp_num_large);
      }
      warp_large_start = __shfl(warp_large_start, 0);
      // if current warp has elements, add elements to the array
      if(thread_prob >= cutoff) {
        large[warp_large_start + lane_offset(warp_large_bits, threadIdx.x)] = i;
      }
    }
  }
  __syncthreads();
  // grab a set of indices from large array for the warp to work on

  // loop over each warp's range

    // try to place probabilities into current window

    // place any small probabilities into slots and small index array, grab new probability

    // if large stack empty, perform a warp rebalance

    // if large stack empty and cannot rebalance, write current index to window stack

  __syncthreads();
  // if still holding indices, grab an index from window stack and iterate over that


  __syncthreads();
  // if window stack empty, grab a set of elements from small stack and iterate over that

  __syncthreads();
  // at this point, all remaining slots must have probability 1
}


__global__ void draw_poisson(float** prob, float** alias, int* lambda, int n) {
}

Poisson::Poisson(int ml, int mv) {
  // assign class parameters
  max_lambda = ml;
  max_value = mv;
  // allocate array of pointers on host first, so cudaMalloc can populate it
  float** prob_host = new float*[max_lambda];
  float** alias_host = new float*[max_lambda];
  // allocate each Alias table
  for(size_t i = 0; i < max_lambda; ++i) {
    cudaMalloc(&prob_host[i], max_value * sizeof(float)) >> GPLDA_CHECK;
    cudaMalloc(&alias_host[i], max_value * sizeof(float)) >> GPLDA_CHECK;
  }
  // now, allocate array of pointers on device
  cudaMalloc(&prob, max_lambda * sizeof(float*)) >> GPLDA_CHECK;
  cudaMalloc(&alias, max_lambda * sizeof(float*)) >> GPLDA_CHECK;
  // copy array of pointers to device
  cudaMemcpy(prob, prob_host, max_lambda * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMemcpy(alias, alias_host, max_lambda * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
  // launch kernel to build the alias tables
  build_poisson<<<max_lambda,64/*32*/,max_value*sizeof(int)>>>(prob, alias, ARGS->beta, max_value);
  cudaDeviceSynchronize();
}

Poisson::~Poisson() {
  // allocate array of pointers on host, so we can dereference it
  float** prob_host = new float*[max_lambda];
  float** alias_host = new float*[max_lambda];
  // copy array of pointers to host
  cudaMemcpy(prob_host, prob, max_lambda * sizeof(float*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(alias_host, alias, max_lambda * sizeof(float*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  // free the memory at the arrays being pointed to
  for(size_t i = 0; i < max_lambda; ++i) {
    cudaFree(prob_host[i]) >> GPLDA_CHECK;
    cudaFree(alias_host[i]) >> GPLDA_CHECK;
  }
  // free the memory of the pointer array on device
  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(alias) >> GPLDA_CHECK;
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
}

}
