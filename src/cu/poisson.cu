#include "poisson.h"
#include "train.h"

namespace gplda {

Poisson::Poisson(uint32_t max_lambda, size_t max_value) {
  // allocate array of pointers on host first, so cudaMalloc can populate it
  float** prob_host = new float*[max_lambda];
  float** alias_host = new float*[max_lambda];
  // allocate each Alias table
  for(int i = 0; i < max_lambda; ++i) {
    cudaMalloc(&prob_host[i], max_value * sizeof(float));
    cudaMalloc(&alias_host[i], max_value * sizeof(float));
  }
  // now, allocate array of pointers on device
  cudaMalloc(&prob, max_lambda*sizeof(float**));
  cudaMalloc(&alias, max_lambda*sizeof(float**));
  // copy array of pointers to device
  cudaMemcpy(&prob, &prob_host, max_lambda*sizeof(float**), cudaMemcpyHostToDevice);
  cudaMemcpy(&alias, &alias_host, max_lambda*sizeof(float**), cudaMemcpyHostToDevice);
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
  // launch kernel to build Poisson alias tables
  build_poisson<<<1,1>>>(prob, alias, ARGS->beta, max_lambda, max_value);
}

Poisson::~Poisson() {
  for(int i = 0; i < 0; ++i) {
    cudaFree(&prob[i]);
    cudaFree(&alias[i]);
  }
}

__global__ void build_poisson(float** prob, float** alias, float beta, uint32_t lambda, size_t size) {
}

__global__ void draw_poisson() {
}

}
