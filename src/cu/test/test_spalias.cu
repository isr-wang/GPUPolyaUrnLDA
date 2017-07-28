#include "test_spalias.cuh"
#include "../spalias.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;

namespace gplda_test {

void test_build_alias() {
  uint32_t table_size = 10;

  float** prob;
  float** alias;

  cudaMalloc(&prob, sizeof(float*)) >> GPLDA_CHECK;
  cudaMalloc(&alias, sizeof(float*)) >> GPLDA_CHECK;

  float** prob_host[1];
  float** alias_host[1];

  float prob_host_values[10] = {0.9,0.02,0.01,0.01,0.01, 0.01,0.01,0.01,0.01,0.01};
  float alias_host_values[10];

  cudaMalloc(&prob_host[0], table_size * sizeof(float)) >> GPLDA_CHECK;
  cudaMalloc(&alias_host[0], table_size * sizeof(float)) >> GPLDA_CHECK;

  cudaMemcpy(prob, prob_host, sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMemcpy(alias, alias_host, sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  cudaMemcpy(prob_host[0], prob_host_values, table_size * sizeof(float), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::build_alias<<<1,64>>>(prob, alias, 10);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(prob_host_values, prob_host[0], table_size * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(alias_host_values, alias_host[0], table_size * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(int32_t i = 0; i < table_size; ++i) {
    assert(prob_host_values[i] <= 0.02f);
    assert(alias_host_values[i] == 0.0f);
  }

  cudaFree(prob_host[0]) >> GPLDA_CHECK;
  cudaFree(alias_host[0]) >> GPLDA_CHECK;

  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(alias) >> GPLDA_CHECK;
}

}
