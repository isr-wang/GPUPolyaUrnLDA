#include "test_spalias.cuh"
#include "../spalias.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;
using gplda::f32;
using gplda::i32;
using gplda::u32;
using gplda::u64;

namespace gplda_test {

void test_build_alias() {
  u32 table_size = 10;

  f32** prob;
  u32** alias;

  cudaMalloc(&prob, sizeof(f32*)) >> GPLDA_CHECK;
  cudaMalloc(&alias, sizeof(u32*)) >> GPLDA_CHECK;

  f32** prob_host[1];
  u32** alias_host[1];

  f32 prob_host_values[10] = {0.9,0.02,0.01,0.01,0.01, 0.01,0.01,0.01,0.01,0.01};
  u32 alias_host_values[10];

  cudaMalloc(&prob_host[0], table_size * sizeof(f32)) >> GPLDA_CHECK;
  cudaMalloc(&alias_host[0], table_size * sizeof(u32)) >> GPLDA_CHECK;

  cudaMemcpy(prob, prob_host, sizeof(f32*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMemcpy(alias, alias_host, sizeof(u32*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  cudaMemcpy(prob_host[0], prob_host_values, table_size * sizeof(f32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::build_alias<<<1,64>>>(prob, alias, 10);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(prob_host_values, prob_host[0], table_size * sizeof(f32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(alias_host_values, alias_host[0], table_size * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < table_size; ++i) {
    assert(prob_host_values[i] <= 0.02f);
    assert(alias_host_values[i] == 0);
  }

  cudaFree(prob_host[0]) >> GPLDA_CHECK;
  cudaFree(alias_host[0]) >> GPLDA_CHECK;

  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(alias) >> GPLDA_CHECK;
}

}
