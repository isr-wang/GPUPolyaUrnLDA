#include "train.h"

__global__ void polya_urn_sampler() {
}

__device__ __forceinline__ int draw_poisson(float beta, int n, float u, int L) {
  return (n < 100) ? 0/*alias*/ : (int) llrintf(normcdfinvf(u)*sqrtf(beta + n) + beta + n);
}
