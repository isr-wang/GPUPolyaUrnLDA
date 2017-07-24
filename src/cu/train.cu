#include "stdint.h"

#include <cuda_runtime.h>
#include <cublas_v2.h> // need to add -lcublas to nvcc flags

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

// externally visible global variables
Args* args;
DSMatrix<float>* Phi;
DSMatrix<uint32_t>* n;
Poisson* pois;
SpAlias* alias;
float* sigma_a;

// other global variables
cudaStream_t* Phi_stream;
cublasHandle_t* cublas_handle;
DSMatrix<float>* Phi_temp;
float* d_one;
float* d_zero;

extern "C" void initialize(Args* init_args, Buffer* buffers, size_t n_buffers) {
  // set the pointer to args struct
  args = init_args;

  // allocate and initialize cuBLAS
  cublas_handle = new cublasHandle_t;
  cublasCreate(cublas_handle) >> GPLDA_CHECK;
  cublasSetPointerMode(*cublas_handle, CUBLAS_POINTER_MODE_DEVICE) >> GPLDA_CHECK;
  cudaMalloc(&d_one, sizeof(float)) >> GPLDA_CHECK;
  cudaMemset(d_one, 1.0f, 1) >> GPLDA_CHECK;
  cudaMalloc(&d_zero, sizeof(float)) >> GPLDA_CHECK;
  cudaMemset(d_zero, 0.0f, 1) >> GPLDA_CHECK;
  Phi_temp = new DSMatrix<float>();

  // allocate and initialize streams
  Phi_stream = new cudaStream_t;
  cudaStreamCreate(Phi_stream) >> GPLDA_CHECK;

  // allocate memory for buffers
  for(size_t i = 0; i < n_buffers; ++i) {
    buffers[i].stream = new cudaStream_t;
    cudaStreamCreate(buffers[i].stream) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_z, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_w, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_len, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_idx, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
  }

  // allocate globals
  Phi = new DSMatrix<float>();
  n = new DSMatrix<uint32_t>();
  pois = new Poisson(POIS_MAX_LAMBDA, POIS_MAX_VALUE);
  alias = new SpAlias(args->V, args->K);
  cudaMalloc(&sigma_a,args->V * sizeof(float)) >> GPLDA_CHECK;
}

extern "C" void cleanup(Buffer* buffers, size_t n_buffers) {
  // deallocate globals
  cudaFree(sigma_a) >> GPLDA_CHECK;
  delete alias;
  delete pois;
  delete n;
  delete Phi;

  // deallocate memory for buffers
  for(size_t i = 0; i < n_buffers; ++i) {
    cudaFree(buffers[i].gpu_z) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_w) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_len) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_idx) >> GPLDA_CHECK;
    cudaStreamDestroy(*buffers[i].stream) >> GPLDA_CHECK;
    delete buffers[i].stream;
  }

  // deallocate streams
  cudaStreamDestroy(*Phi_stream) >> GPLDA_CHECK;
  delete Phi_stream;

  // deallocate cuBLAS
  delete Phi_temp;
  cudaFree(d_zero) >> GPLDA_CHECK;
  cudaFree(d_one) >> GPLDA_CHECK;
  cublasDestroy(*cublas_handle) >> GPLDA_CHECK;
  delete cublas_handle;

  // remove the args pointer
  args = NULL;
}

extern "C" void sample_phi() {
  // draw Phi ~ PPU(n + beta)
  polya_urn_sample<<<args->K,256,0,*Phi_stream>>>(Phi->dense, n->dense, args->beta, args->V, alias->prob, alias->alias);

  // copy Phi for transpose, set the stream, then transpose Phi
  cudaMemcpyAsync(Phi_temp->dense, Phi->dense, args->V * args->K * sizeof(float), cudaMemcpyDeviceToDevice, *Phi_stream) >> GPLDA_CHECK;
  cublasSetStream(*cublas_handle, *Phi_stream) >> GPLDA_CHECK; //
  cublasSgeam(*cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, args->K, args->V, d_one, Phi_temp->dense, args->V, d_zero, Phi->dense, args->K, Phi->dense, args->K) >> GPLDA_CHECK;

  // compute sigma_a and alias probabilities
  polya_urn_colsums<<<args->V,128,0,*Phi_stream>>>(Phi->dense, sigma_a, alias->prob, args->K);

  // build Alias tables
  build_alias<<<args->V,32,2*next_pow2(args->K)*sizeof(int), *Phi_stream>>>(alias->prob, alias->alias, args->K);

  // reset sufficient statistics for n
  cudaMemsetAsync(n->dense, 0, args->K * args->V, *Phi_stream) >> GPLDA_CHECK;

  // don't return until operations completed
  cudaStreamSynchronize(*Phi_stream) >> GPLDA_CHECK;
}

extern "C" void sample_z_async(Buffer* buffer) {
  // copy z,w,d to GPU and compute d_idx based on document length
  cudaMemcpyAsync(buffer->gpu_z, buffer->z, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy z to GPU
  cudaMemcpyAsync(buffer->gpu_w, buffer->w, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy w to GPU
  cudaMemcpyAsync(buffer->gpu_d_len, buffer->d, buffer->n_docs, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK;
  compute_d_idx<<<buffer->n_docs,32,0,*buffer->stream>>>(buffer->gpu_d_len, buffer->gpu_d_idx, buffer->n_docs);

  // sample the topic indicators
  warp_sample_topics<<<buffer->n_docs,32,0,*buffer->stream>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);

  // copy z back to host
  cudaMemcpyAsync(buffer->z, buffer->gpu_z, buffer->size, cudaMemcpyDeviceToHost,*buffer->stream) >> GPLDA_CHECK;
}

extern "C" void sync_buffer(Buffer *buffer) {
  // return when stream has finished
  cudaStreamSynchronize(*buffer->stream) >> GPLDA_CHECK;
}

}
