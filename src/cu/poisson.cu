#include "poisson.h"
#include "train.h"

namespace gplda {

__global__ void build_poisson(float** prob, float** alias, float beta, uint32_t lambda, size_t size) {
}

__global__ void draw_poisson(float** prob, float** alias, uint32_t* lambda, size_t n) {
}

Poisson::Poisson(uint32_t ml, size_t mv) {
  // assign class parameters
  max_lambda = ml;
  max_value = mv;
  // allocate array of pointers on host first, so cudaMalloc can populate it
  float** prob_host = new float*[max_lambda];
  float** alias_host = new float*[max_lambda];
  // allocate each Alias table
  for(int i = 0; i < max_lambda; ++i) {
    cudaMalloc(&prob_host[i], max_value * sizeof(float));
    cudaMalloc(&alias_host[i], max_value * sizeof(float));
  }
  // now, allocate array of pointers on device
  cudaMalloc(&prob, max_lambda * sizeof(float**));
  cudaMalloc(&alias, max_lambda * sizeof(float**));
  // copy array of pointers to device
  cudaMemcpy(&prob, &prob_host, max_lambda * sizeof(float**), cudaMemcpyHostToDevice);
  cudaMemcpy(&alias, &alias_host, max_lambda * sizeof(float**), cudaMemcpyHostToDevice);
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
  cudaMemcpy(&prob_host, &prob, max_lambda * sizeof(float**), cudaMemcpyDeviceToHost);
  cudaMemcpy(&alias_host, &alias, max_lambda * sizeof(float**), cudaMemcpyDeviceToHost);
  // free the memory at the arrays being pointed to
  for(int i = 0; i < max_value; ++i) {
    cudaFree(&prob_host[i]);
    cudaFree(&alias_host[i]);
  }
  // free the memory of the pointer array on device
  cudaFree(&prob);
  cudaFree(&alias);
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
}

}
