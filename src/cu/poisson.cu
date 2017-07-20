#include "error.cuh"
#include "poisson.cuh"
#include "spalias.cuh"
#include "train.cuh"


namespace gplda {

__global__ void build_poisson_prob(float** prob, float** alias, float beta, int table_size) {
  // determine constants
  float lambda = blockIdx.x + beta; // each block builds one table
  // populate PMF
  for(int offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int i = threadIdx.x + offset * blockDim.x;
    float x = i;
    if(i < table_size) {
      prob[blockIdx.x][i] = expf(x*logf(lambda) - lambda - lgammaf(x + 1));
    }
  }
}

__global__ void draw_poisson(float** prob, float** alias, int* lambda, int n) {
}

inline int next_pow2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return(x);
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
  build_poisson_prob<<<max_lambda,96>>>(prob, alias, ARGS->beta, max_value);
  cudaDeviceSynchronize() >> GPLDA_CHECK;
  build_alias<<<max_lambda,32,2*next_pow2(max_value)*sizeof(int)>>>(prob, alias, max_value);
  cudaDeviceSynchronize() >> GPLDA_CHECK;
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
