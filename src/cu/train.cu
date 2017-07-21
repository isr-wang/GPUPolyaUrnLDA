#include "stdint.h"

#include "train.cuh"
#include "dsmatrix.cuh"
#include "error.cuh"
#include "poisson.cuh"
#include "polyaurn.cuh"
#include "spalias.cuh"
#include "warpsample.cuh"

#define POIS_MAX_LAMBDA 100
#define POIS_MAX_VALUE 200
#define DS_DENSE 100
#define DS_SPARSE 1000

namespace gplda {

Args* ARGS;
DSMatrix<float>* Phi;
DSMatrix<uint32_t>* n;
Poisson* pois;
SpAlias* alias;
float* sigma_a;
cudaStream_t* PhiStream;

extern "C" void initialize(Args* args, Buffer* buffers, size_t n_buffers) {
  ARGS = args;
  for(size_t i = 0; i < n_buffers; ++i) {
    buffers[i].stream = new cudaStream_t;
    cudaStreamCreate(buffers[i].stream) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_z, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_w, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_len, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_idx, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
  }
  PhiStream = new cudaStream_t;
  cudaStreamCreate(PhiStream) >> GPLDA_CHECK;
  Phi = new DSMatrix<float>();
  n = new DSMatrix<uint32_t>();
  pois = new Poisson(POIS_MAX_LAMBDA, POIS_MAX_VALUE);
  alias = new SpAlias(ARGS->V, ARGS->K);
  cudaMalloc(&sigma_a,ARGS->V * sizeof(float)) >> GPLDA_CHECK;
}

extern "C" void cleanup(Buffer* buffers, size_t n_buffers) {
  cudaFree(sigma_a) >> GPLDA_CHECK;
  delete alias;
  delete pois;
  delete n;
  delete Phi;
  cudaStreamDestroy(*PhiStream) >> GPLDA_CHECK;
  delete PhiStream;
  for(size_t i = 0; i < n_buffers; ++i) {
    cudaStreamDestroy(*buffers[i].stream) >> GPLDA_CHECK;
    delete buffers[i].stream;
    cudaFree(buffers[i].gpu_z) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_w) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_len) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_idx) >> GPLDA_CHECK;
  }
  ARGS = NULL;
}

extern "C" void sample_phi() {
  polya_urn_sample<<<1,1>>>(); // draw Phi ~ Pois(n + beta)
  polya_urn_normalize<<<1,1>>>(); // normalize to get Phi ~ PPU(n + beta)
  // transpose Phi
  polya_urn_colsums<<<1,1>>>(); // compute sigma_a
//  polya_urn_prob<<<1,1>>>(); // compute probabilities for use in Alias table
//  build_alias<<<1,1>>>(); // build Alias table
  // reset_sufficient_statistics<<<1,1>>>; // reset sufficient statistics for n
}

extern "C" void sample_z_async(Buffer* buffer) {
  cudaMemcpyAsync(buffer->gpu_d_len, buffer->d, buffer->n_docs, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy d to GPU
  compute_d_idx<<<1,1,0,*buffer->stream>>>(buffer->gpu_d_len, buffer->gpu_d_idx, buffer->n_docs);
  cudaMemcpyAsync(buffer->gpu_z, buffer->z, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy z to GPU
  cudaMemcpyAsync(buffer->gpu_w, buffer->w, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy w to GPU
  warp_sample<<<1,1,0,*buffer->stream>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
  cudaMemcpyAsync(buffer->z, buffer->gpu_z, buffer->size, cudaMemcpyDeviceToHost,*buffer->stream) >> GPLDA_CHECK; // copy z back to host
}

extern "C" void sync_buffer(Buffer *buffer) {
  cudaStreamSynchronize(*buffer->stream) >> GPLDA_CHECK;
}

}
