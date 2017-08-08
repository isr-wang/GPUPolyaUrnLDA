#include <cuda_runtime.h>
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include <cublas_v2.h> // need to add -lcublas to nvcc flags

#include "train.cuh"
#include "dsmatrix.cuh"
#include "error.cuh"
#include "poisson.cuh"
#include "polyaurn.cuh"
#include "random.cuh"
#include "spalias.cuh"
#include "warpsample.cuh"

#define POIS_MAX_LAMBDA 100
#define POIS_MAX_VALUE 200
#define DS_DENSE 100
#define DS_SPARSE 1000

namespace gplda {

// global variables
Args* args; // externally visible
DSMatrix<float>* Phi;
DSMatrix<uint32_t>* n;
Poisson* pois;
SpAlias* alias;
float* sigma_a;
uint32_t* C;
curandStatePhilox4_32_10_t* Phi_rng;
cudaStream_t* Phi_stream;
cublasHandle_t* cublas_handle;
DSMatrix<float>* Phi_temp;
float* d_one;
float* d_zero;

extern "C" void initialize(Args* init_args, Buffer* buffers, uint32_t n_buffers) {
  // set the pointer to args struct
  args = init_args;

  // allocate and initialize cuBLAS
  cublas_handle = new cublasHandle_t;
  cublasCreate(cublas_handle) >> GPLDA_CHECK;
  cublasSetPointerMode(*cublas_handle, CUBLAS_POINTER_MODE_DEVICE) >> GPLDA_CHECK;
  float h_zero = 0.0f;
  cudaMalloc(&d_zero, sizeof(float)) >> GPLDA_CHECK;
  cudaMemcpy(d_zero, &h_zero, sizeof(float), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  float h_one = 1.0f;
  cudaMalloc(&d_one, sizeof(float)) >> GPLDA_CHECK;
  cudaMemcpy(d_one, &h_one, sizeof(float), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  Phi_temp = new DSMatrix<float>();

  // allocate and initialize cuRAND
  cudaMalloc(&Phi_rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  rand_init<<<1,1>>>(0, 0, Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  // allocate and initialize streams
  Phi_stream = new cudaStream_t;
  cudaStreamCreate(Phi_stream) >> GPLDA_CHECK;

  // allocate memory for buffers
  for(int32_t i = 0; i < n_buffers; ++i) {
    buffers[i].stream = new cudaStream_t;
    cudaStreamCreate(buffers[i].stream) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_z, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_w, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_len, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_d_idx, buffers[i].size * sizeof(uint32_t)) >> GPLDA_CHECK;
    cudaMalloc(&buffers[i].gpu_rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
    rand_init<<<1,1>>>(0, i + 1, buffers[i].gpu_rng);
    cudaDeviceSynchronize() >> GPLDA_CHECK;
  }

  // allocate globals
  Phi = new DSMatrix<float>();
  n = new DSMatrix<uint32_t>();
  pois = new Poisson(POIS_MAX_LAMBDA, POIS_MAX_VALUE, args->beta);
  alias = new SpAlias(args->V, args->K);
  cudaMalloc(&sigma_a,args->V * sizeof(float)) >> GPLDA_CHECK;
  cudaMalloc(&C,args->V * sizeof(uint32_t)) >> GPLDA_CHECK;
  cudaMemcpy(C, args->C, args->V * sizeof(uint32_t), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  // run device init code
  polya_urn_init<<<args->K,256>>>(n->dense, C, args->beta, args->V, pois->pois_alias->prob, pois->pois_alias->alias, pois->max_lambda, pois->max_value, Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;
  rand_advance<<<1,1>>>(args->K*args->V,Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;
}

extern "C" void cleanup(Buffer* buffers, uint32_t n_buffers) {
  // deallocate globals
  cudaFree(C) >> GPLDA_CHECK;
  cudaFree(sigma_a) >> GPLDA_CHECK;
  delete alias;
  delete pois;
  delete n;
  delete Phi;

  // deallocate memory for buffers
  for(int32_t i = 0; i < n_buffers; ++i) {
    cudaFree(buffers[i].gpu_z) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_w) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_len) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_d_idx) >> GPLDA_CHECK;
    cudaFree(buffers[i].gpu_rng) >> GPLDA_CHECK;
    cudaStreamDestroy(*buffers[i].stream) >> GPLDA_CHECK;
    delete buffers[i].stream;
  }

  // deallocate streams
  cudaStreamDestroy(*Phi_stream) >> GPLDA_CHECK;
  delete Phi_stream;

  // deallocate cuRAND
  cudaFree(Phi_rng) >> GPLDA_CHECK;

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
  polya_urn_sample<<<args->K,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM,0,*Phi_stream>>>(Phi->dense, n->dense, args->beta, args->V, pois->pois_alias->prob, pois->pois_alias->alias, pois->max_lambda, pois->max_value, Phi_rng);
  rand_advance<<<1,1,0,*Phi_stream>>>(args->K*args->V,Phi_rng);

  // copy Phi for transpose, set the stream, then transpose Phi
  polya_urn_transpose(Phi_stream, Phi->dense, Phi_temp->dense, args->K, args->V, cublas_handle, d_zero, d_one);

  // compute sigma_a and alias probabilities
  polya_urn_colsums<<<args->V,GPLDA_POLYA_URN_COLSUMS_BLOCKDIM,0,*Phi_stream>>>(Phi->dense, sigma_a, args->alpha, alias->prob, args->K);

  // build Alias tables
  build_alias<<<args->V,32,2*next_pow2(args->K)*sizeof(int32_t), *Phi_stream>>>(alias->prob, alias->alias, args->K);

  // reset sufficient statistics for n
  polya_urn_reset<<<args->K,128,0,*Phi_stream>>>(n->dense, args->V);

  // don't return until operations completed
  cudaStreamSynchronize(*Phi_stream) >> GPLDA_CHECK;
}

extern "C" void sample_z_async(Buffer* buffer) {
  // copy z,w,d to GPU and compute d_idx based on document length
  cudaMemcpyAsync(buffer->gpu_z, buffer->z, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy z to GPU
  cudaMemcpyAsync(buffer->gpu_w, buffer->w, buffer->size, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK; // copy w to GPU
  cudaMemcpyAsync(buffer->gpu_d_len, buffer->d, buffer->n_docs, cudaMemcpyHostToDevice,*buffer->stream) >> GPLDA_CHECK;
  compute_d_idx<<<1,GPLDA_COMPUTE_D_IDX_BLOCKDIM,0,*buffer->stream>>>(buffer->gpu_d_len, buffer->gpu_d_idx, buffer->n_docs);

  // sample the topic indicators
  warp_sample_topics<<<buffer->n_docs,32,0,*buffer->stream>>>(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx, alias->prob, alias->alias, buffer->gpu_rng);
  rand_advance<<<1,1,0,*buffer->stream>>>(buffer->size,Phi_rng);

  // copy z back to host
  cudaMemcpyAsync(buffer->z, buffer->gpu_z, buffer->size, cudaMemcpyDeviceToHost,*buffer->stream) >> GPLDA_CHECK;
}

extern "C" void sync_buffer(Buffer *buffer) {
  // return when stream has finished
  cudaStreamSynchronize(*buffer->stream) >> GPLDA_CHECK;
}

}
