#include "error.cuh"
#include "poisson.cuh"
#include "spalias.cuh"
#include "train.cuh"


namespace gplda {

__global__ void build_poisson_prob(float** prob, float beta, uint32_t table_size) {
  // determine constants
  float lambda = blockIdx.x + beta; // each block builds one table
  // populate PMF
  for(int32_t offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int32_t i = threadIdx.x + offset * blockDim.x;
    float x = i;
    if(i < table_size) {
      prob[blockIdx.x][i] = expf(x*logf(lambda) - lambda - lgammaf(x + 1));
    }
  }
}

Poisson::Poisson(uint32_t ml, uint32_t mv) {
  // assign class parameters
  max_lambda = ml;
  max_value = mv;
  // allocate alias table
  pois_alias = new SpAlias(max_lambda, max_value);
  // launch kernel to build the alias tables
  build_poisson_prob<<<max_lambda,96>>>(pois_alias->prob, args->beta, max_value);
  cudaDeviceSynchronize() >> GPLDA_CHECK;
  build_alias<<<max_lambda,32,2*next_pow2(max_value)*sizeof(int32_t)>>>(pois_alias->prob, pois_alias->alias, max_value);
  cudaDeviceSynchronize() >> GPLDA_CHECK;
}

Poisson::~Poisson() {
  delete pois_alias;
}

}
