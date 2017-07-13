#include "train.h"
#include "dsmatrix.h"
#include "error.h"
#include "poisson.h"
#include "polyaurn.h"
#include "spalias.h"
#include "warpsample.h"

#define POIS_MAX_LAMBDA 100
#define POIS_MAX_VALUE 200
#define DLH_DENSE 100
#define DLH_SPARSE 1000

namespace gplda {

Args* ARGS;
DSMatrix<float>* Phi;
DSMatrix<uint32_t>* n;
Poisson* pois;
SpAlias* alias;
float* sigma_a;

extern "C" void initialize(Args* args, Buffer* buffers, size_t n_buffers) {
  ARGS = args;
  for(size_t i = 0; i < n_buffers; ++i) {
    cudaMalloc(&buffers[i].gpu_z, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_w, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_len, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_idx, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
  }
  Phi = new DSMatrix<float>();
  n = new DSMatrix<uint32_t>();
  pois = new Poisson(POIS_MAX_LAMBDA, POIS_MAX_VALUE);
  alias = new SpAlias();
  cudaMalloc(&sigma_a,ARGS->V * sizeof(float)) >> GPLDA_CHECK;
}

extern "C" void cleanup(Buffer* buffers, size_t n_buffers) {
  cudaFree(sigma_a) >> GPLDA_CHECK;
  delete alias;
  delete pois;
  delete n;
  delete Phi;
  for(size_t i = 0; i < n_buffers; ++i) {
    cudaFree(buffers[i].gpu_z) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_w) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_len) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_idx) >> GPLDA_CHECK;
  }
  ARGS = NULL;
}

extern "C" void sample_phi() {
  polya_urn_sample<<<1,1>>>();
  polya_urn_normalize<<<1,1>>>();
  // transpose Phi
  polya_urn_colsums<<<1,1>>>();
  build_alias<<<1,1>>>();
}

extern "C" void sample_z(Buffer* buffer) {
  // copy memory
  warp_sample<<<1,1>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
}

extern "C" void sync_buffer(Buffer *buffer) {
//  cudaStreamSynchronize(*buffer->stream);
}

}
