#include "polyaurn.cuh"
#include "poisson.cuh"

namespace gplda {

__device__ __forceinline__ float draw_poisson(float beta, uint32_t n, float** prob, float** alias) {
  // MUST be defined in this file to compile on all platforms
  return beta;
}

__global__ void polya_urn_sample(float* Phi, uint32_t* n, float beta, uint32_t V, float** prob, float** alias) {
  // loop over array and draw samples
  float sum = 0.0f;
  for(int offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int col = threadIdx.x + offset * blockDim.x;
    int i = col + V * blockIdx.x;
    if(col < V) {
      float pois = draw_poisson(beta, n[i], prob, alias);
      Phi[i] = pois;
      sum += pois;
    }
  }
  __syncthreads();
  // normalize draws
  for(int offset = 0; offset < V / blockDim.x + 1; ++offset) {
    int col = threadIdx.x + offset * blockDim.x;
    int i = col + V * blockIdx.x;
    if(col < V) {
      Phi[i] /= sum;
    }
  }
}

__global__ void polya_urn_colsums(float* Phi, float* sigma_a, float** prob, uint32_t K) {

}

}
