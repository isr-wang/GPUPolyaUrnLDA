#ifndef GPLDA_TRAIN_H
#define GPLDA_TRAIN_H

#include "stdint.h"
#include "dlhmatrix.h"
#include "poisson.h"
#include "spalias.h"

namespace gplda {

struct Args {
  float alpha;
  float beta;
  uint32_t K;
  uint32_t L;
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
};

extern Args* ARGS;
extern DLHMatrix* Phi;
extern DLHMatrix* n;
extern Poisson* pois;
extern SpAlias* alias;

extern "C" void initialize(Args* args, Buffer* buffers, size_t n_buffers);
extern "C" void sample_phi();
extern "C" void sample_z(Buffer* buffer);
extern "C" void cleanup(Buffer* buffers, size_t n_buffers);

}

#endif
