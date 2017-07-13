#ifndef GPLDA_TRAIN_H
#define GPLDA_TRAIN_H

#include "stdint.h"
#include "dsmatrix.h"
#include "poisson.h"
#include "spalias.h"

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
  uint32_t* d_len;
  uint32_t* d_idx;
  size_t n_docs;
  uint32_t* gpu_z;
  uint32_t* gpu_w;
  uint32_t* gpu_d_len;
  uint32_t* gpu_d_idx;
  cudaStream_t* stream;
};

extern Args* ARGS;
extern DSMatrix<float>* Phi;
extern DSMatrix<uint32_t>* n;
extern Poisson* pois;
extern SpAlias* alias;
extern float* sigma_a;

extern "C" void initialize(Args* args, Buffer* buffers, size_t n_buffers);
extern "C" void sample_phi();
extern "C" void sample_z(Buffer* buffer);
extern "C" void cleanup(Buffer* buffers, size_t n_buffers);
extern "C" void sync_buffer(Buffer* buffer);

}

#endif
