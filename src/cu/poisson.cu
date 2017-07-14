#include "assert.h"
#include "error.cuh"
#include "poisson.cuh"
#include "train.cuh"

namespace gplda {

__global__ void build_poisson(float** prob, float** alias, float beta, int table_size) {
  assert(blockDim.x == 32); // for simplicity, Poisson Alias tables are built on the warp level, so exit if misconfigured
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
  extern __shared__ float large[];
  __shared__ int num_large[1];
  float cutoff = 1.0/((float) table_size);
  // loop over PMF
  for(int offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int i = threadIdx.x + offset * blockDim.x;
    // determine which warps have large probabilities
    unsigned int warp_large = __ballot(prob[lambda][i] > cutoff);
    // determine how many large probabilities are in the warp's view
    int warp_num_large = __popc(warp_large);
    // increment the array's size
    int large_start = atomicAdd(num_large, warp_num_large);
    // if current warp has elements, add elements to the array
    if(1/*warp_bit_set*/) {
      large[large_start + 0 /*warp_bit_offset*/] = prob[lambda][i];
    }
  }
  // we've now built large array, let's grab elements and place them

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
  build_poisson<<<max_lambda,32>>>(prob, alias, ARGS->beta, max_value);
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
