#include <stdlib.h>
#include <iostream>
#include "types.cuh"
#include "train.cuh"
#include "test/run_tests.cuh"


using gplda::f32;
using gplda::i32;
using gplda::u32;
using gplda::u64;

int main(void) {
  gplda_test::run_tests();

  constexpr f32 alpha = 0.1;
  constexpr f32 beta = 0.1;
  constexpr u32 V = 5;
  constexpr u32 K = 10;
  u32 C[V] = {1,1,1,1,1};
  constexpr u32 buffer_size = 5;
  constexpr u32 buffer_max_docs = 2;

  gplda::Args args = {alpha,beta,K,V,C,buffer_size,buffer_max_docs};
  u32 z[buffer_size] = {0,0,0,0,0};
  u32 w[buffer_size] = {0,1,2,3,4};
  u32 d[buffer_max_docs] = {3,2};
  u32 K_d[buffer_max_docs] = {1,1};
  u32 n_docs = buffer_max_docs;
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
