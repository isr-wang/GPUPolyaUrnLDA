#include "train.h"
#include "alias.h"
#include "dlhmatrix.h"
#include "poisson.h"
#include "polyaurnsampler.h"
#include "warpsampler.h"

namespace gplda {

Args *ARGS;
DLHMatrix *Phi;
DLHMatrix *n;
Poisson *pois;

extern "C" void initialize(Args *args, Buffer *buffers, size_t n_buffers) {
  ARGS = args;
  for(int i = 0; i < n_buffers; ++i) {
    cudaMalloc((void**)&buffers[i].gpu_z, buffers[i].size * sizeof(uint32_t));
    cudaMalloc((void**)&buffers[i].gpu_w, buffers[i].size * sizeof(uint32_t));
    cudaMalloc((void**)&buffers[i].gpu_d_len, buffers[i].size * sizeof(uint32_t));
    cudaMalloc((void**)&buffers[i].gpu_d_idx, buffers[i].size * sizeof(uint32_t));
  }
  Phi = new DLHMatrix();
  n = new DLHMatrix();
  pois = new Poisson();
}

extern "C" void cleanup(Buffer *buffers, size_t n_buffers) {
  ARGS = NULL;
  for(int i = 0; i < n_buffers; ++i) {
    cudaFree(buffers[i].gpu_z);
    cudaFree(buffers[i].gpu_w);
    cudaFree(buffers[i].gpu_d_len);
    cudaFree(buffers[i].gpu_d_idx);
  }
  delete Phi;
  delete n;
  delete pois;
}

extern "C" void sample_phi() {
  polya_urn_sampler<<<1,1>>>();
}

extern "C" void sample_z(Buffer *buffer) {
  // copy memory
  warp_sampler<<<1,1>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
}

}
