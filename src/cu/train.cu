#include "train.h"
#include "dlhmatrix.h"
#include "poisson.h"
#include "polyaurnsampler.h"
#include "spalias.h"
#include "warpsampler.h"

#define POIS_MAX_LAMBDA 100
#define POIS_MAX_VALUE 200

namespace gplda {

Args* ARGS;
DLHMatrix* Phi;
DLHMatrix* n;
Poisson* pois;
SpAlias* alias;

extern "C" void initialize(Args* args, Buffer* buffers, size_t n_buffers) {
  ARGS = args;
  for(int i = 0; i < n_buffers; ++i) {
    cudaMalloc(&buffers[i].gpu_z, buffers[i].size * sizeof(uint32_t));
    cudaMalloc(&buffers[i].gpu_w, buffers[i].size * sizeof(uint32_t));
    cudaMalloc(&buffers[i].gpu_d_len, buffers[i].size * sizeof(uint32_t));
    cudaMalloc(&buffers[i].gpu_d_idx, buffers[i].size * sizeof(uint32_t));
  }
  Phi = new DLHMatrix();
  n = new DLHMatrix();
  pois = new Poisson(POIS_MAX_LAMBDA, POIS_MAX_VALUE);
  alias = new SpAlias();
}

extern "C" void cleanup(Buffer *buffers, size_t n_buffers) {
  delete alias;
  delete pois;
  delete n;
  delete Phi;
  for(int i = 0; i < n_buffers; ++i) {
    cudaFree(buffers[i].gpu_z);
    cudaFree(buffers[i].gpu_w);
    cudaFree(buffers[i].gpu_d_len);
    cudaFree(buffers[i].gpu_d_idx);
  }
  ARGS = NULL;
}

extern "C" void sample_phi() {
  polya_urn_sampler<<<1,1>>>();
  build_alias<<<1,1>>>();
}

extern "C" void sample_z(Buffer *buffer) {
  // copy memory
  warp_sampler<<<1,1>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
}

}
