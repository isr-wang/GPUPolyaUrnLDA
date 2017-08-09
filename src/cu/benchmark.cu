#include <stdlib.h>
#include <iostream>
#include "stdint.h"
#include "train.cuh"
#include "test/run_tests.cuh"

int main(void) {
  gplda_test::run_tests();

  constexpr float alpha = 0.1;
  constexpr float beta = 0.1;
  constexpr uint32_t V = 5;
  constexpr uint32_t K = 10;
  uint32_t C[V] = {1,1,1,1,1};
  constexpr uint32_t buffer_size = 5;
  constexpr uint32_t buffer_max_docs = 2;

  gplda::Args args = {alpha,beta,K,V,C,buffer_size,buffer_max_docs};
  uint32_t z[buffer_size] = {0,0,0,0,0};
  uint32_t w[buffer_size] = {0,1,2,3,4};
  uint32_t d[buffer_max_docs] = {3,2};
  uint32_t K_d[buffer_max_docs] = {1,1};
  uint32_t n_docs = buffer_max_docs;
  gplda::Buffer buffer = {z, w, d, K_d, n_docs, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

  std::cout << "initializing" << std::endl;
  gplda::initialize(&args, &buffer, 1);

  std::cout << "launching poisson polya urn sampler" << std::endl;
  gplda::sample_phi();

  std::cout << "launching warp sampler" << std::endl;
  gplda::sample_z_async(&buffer);
  gplda::sync_buffer(&buffer);

  std::cout << "cleanup" << std::endl;
  gplda::cleanup(&buffer, 1);

  std::cout << "finished" << std::endl;
  return 0;
}
