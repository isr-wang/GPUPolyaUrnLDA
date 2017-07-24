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

Args* args;
DSMatrix<float>* Phi;
DSMatrix<uint32_t>* n;
Poisson* pois;
SpAlias* alias;
float* sigma_a;
cudaStream_t* PhiStream;

extern "C" void initialize(Args* init_args, Buffer* buffers, size_t n_buffers) {
  args = init_args;
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
  alias = new SpAlias(args->V, args->K);
  cudaMalloc(&sigma_a,args->V * sizeof(float)) >> GPLDA_CHECK;
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
    cudaFree(buffers[i].gpu_z) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_w) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_len) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_idx) >> GPLDA_CHECK;
    cudaStreamDestroy(*buffers[i].stream) >> GPLDA_CHECK;
    delete buffers[i].stream;
  }
  args = NULL;
}

extern "C" void sample_phi() {
  polya_urn_sample<<<args->K,256>>>(Phi->dense, n->dense, args->beta, args->V, alias->prob, alias->alias); // draw Phi ~ PPU(n + beta)
  polya_urn_transpose<<<1,1>>>(Phi->dense); // transpose Phi
  polya_urn_colsums<<<args->V,128>>>(Phi->dense, sigma_a, alias->prob, args->K); // compute sigma_a and alias probabilities
  build_alias<<<args->V,32,2*next_pow2(args->K)*sizeof(int)>>>(alias->prob, alias->alias, args->K); // build Alias table
  cudaMemset(n->dense, 0, args->K * args->V); // reset sufficient statistics for n
}

extern "C" void sample_z_async(Buffer* buffer) {
  cudaMemcpyAsync(buffer->gpu_d_len, buffer->d, buffer->n_docs, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy d to GPU
  compute_d_idx<<<buffer->n_docs,32,0,*buffer->stream>>>(buffer->gpu_d_len, buffer->gpu_d_idx, buffer->n_docs);
  cudaMemcpyAsync(buffer->gpu_z, buffer->z, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy z to GPU
  cudaMemcpyAsync(buffer->gpu_w, buffer->w, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy w to GPU
  warp_sample_topics<<<buffer->n_docs,32,0,*buffer->stream>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
  cudaMemcpyAsync(buffer->z, buffer->gpu_z, buffer->size, cudaMemcpyDeviceToHost,*buffer->stream) >> GPLDA_CHECK; // copy z back to host
}

extern "C" void sync_buffer(Buffer *buffer) {
  cudaStreamSynchronize(*buffer->stream) >> GPLDA_CHECK;
}

}
