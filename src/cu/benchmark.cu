#include <stdlib.h>
#include <iostream>
#include "stdint.h"
#include "train.cuh"

int main(void) {
  gplda::Args args = {0.1,0.1,10,5};
  uint32_t z[5] = {0,0,0,0,0};
  uint32_t w[5] = {0,1,2,3,4};
  uint32_t d_len[5] = {3,2,0,0,0};
  uint32_t d_idx[5] = {0,3,0,0,0};
  gplda::Buffer buffer = {5, z, w, d_len, d_idx, 2, NULL, NULL, NULL, NULL, /*cudaStream_t*/ NULL};

  std::cout << "initializing" << std::endl;
  gplda::initialize(&args, &buffer, 1);

  std::cout << "launching poisson polya urn sampler" << std::endl;
  gplda::sample_phi();

  std::cout << "launching warp sampler" << std::endl;
  gplda::sample_z(&buffer);

  std::cout << "cleanup" << std::endl;
  gplda::cleanup(&buffer, 1);

  std::cout << "finished" << std::endl;
  return 0;
}
