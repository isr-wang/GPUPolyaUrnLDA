#include "error.h"
#include "poisson.h"
#include "train.h"

namespace gplda {

__global__ void build_poisson(float** prob, float** alias, float beta, size_t lambda, size_t size) {
}

__global__ void draw_poisson(float** prob, float** alias, size_t* lambda, size_t n) {
}

Poisson::Poisson(size_t ml, size_t mv) {
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
  build_poisson<<<max_lambda,1>>>(prob, alias, ARGS->beta, max_lambda, max_value);
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
