#include "random.cuh"

namespace gplda {

// initializer for random number generator
__global__ void rng_init(u32 seed, u32 subsequence, curandStatePhilox4_32_10_t* rng) {
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init((unsigned long long) seed, (unsigned long long) subsequence, (unsigned long long) 0, rng);
  }
}

// advance for random number generator
__global__ void rng_advance(u32 advance, curandStatePhilox4_32_10_t* rng) {
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    skipahead((unsigned long long) advance, rng);
  }
}

}
