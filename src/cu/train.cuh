#ifndef GPLDA_TRAIN_H
#define GPLDA_TRAIN_H

#include "stdint.h"

#include <cuda_runtime.h>
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#include "dsmatrix.cuh"
#include "poisson.cuh"
#include "spalias.cuh"

namespace gplda {

struct Args {
  float alpha;
  float beta;
  uint32_t K;
  uint32_t V;
};

struct Buffer {
  size_t size;
  uint32_t* z;
  uint32_t* w;
  uint32_t* d;
  size_t n_docs;
  uint32_t* gpu_z;
  uint32_t* gpu_w;
  uint32_t* gpu_d_len;
  uint32_t* gpu_d_idx;
  curandStatePhilox4_32_10_t* gpu_rng;
  cudaStream_t* stream;
};

extern Args* args;
extern DSMatrix<float>* Phi;
extern DSMatrix<uint32_t>* n;
extern Poisson* pois;
extern SpAlias* alias;
extern float* sigma_a;

extern "C" void initialize(Args* args, Buffer* buffers, size_t n_buffers);
extern "C" void sample_phi();
extern "C" void sample_z_async(Buffer* buffer);
extern "C" void cleanup(Buffer* buffers, size_t n_buffers);
extern "C" void sync_buffer(Buffer* buffer);

}

#endif
