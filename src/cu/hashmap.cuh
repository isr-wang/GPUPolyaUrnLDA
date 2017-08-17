#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#define GPLDA_HASH_EMPTY 0xffffffff00000000

namespace gplda {

struct HashMap {
  u32 size;
  u32 num_elements;
  u64* data;
  u64* stash;
  u64* temp_data;
  u64* temp_stash;
  u32 a[GPLDA_HASH_NUM_FUNCTIONS];
  u32 b[GPLDA_HASH_NUM_FUNCTIONS];
  u32 a_stash;
  u32 b_stash;

  __device__ __forceinline__ u32 left_32_bits(u64 x) {
    return (u32) (x >> 32);
  }

  __device__ __forceinline__ u32 right_32_bits(u64 x) {
    return (u32) x;
  }

  __device__ __forceinline__ i32 hash_fn(i32 k, i32 a, i32 b, i32 s) {
    return ((a * k + b) % 334214459) % s;
  }

  __device__ __forceinline__ i32 hash_idx(u32 key, i32 slot, u32* a, u32* b, u32 size) {
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
      i32 possible_slot = hash_fn(key, a[i], b[i], size);
      if(possible_slot == slot) {
        return i;
      }
    }
    return 0;
  }

  __device__ inline void init(void* temp, u32 size, i32 dim, curandStatePhilox4_32_10_t* rng) {
    // determine constants
    i32 thread_idx = threadIdx.x % dim;

    // set map parameters and calculate random hash functions
    this->size = size;
    this->num_elements = 0;
    this->data = (u64*) temp;
    this->stash = this->data + size*sizeof(u64);
    this->temp_data = this->stash + GPLDA_HASH_STASH_SIZE*sizeof(u64);
    this->temp_stash = this->temp_data + size*sizeof(u64);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
      this->a[i] = __float2uint_rz(size * curand_uniform(rng));
      this->b[i] = __float2uint_rz(size * curand_uniform(rng));
    }
    this->a_stash = __float2uint_rz(size * curand_uniform(rng));
    this->b_stash = __float2uint_rz(size * curand_uniform(rng));

    // set map to empty
    for(i32 offset = 0; offset < size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        this->data[i] = GPLDA_HASH_EMPTY;
      }
    }

    // set stash to empty
    #pragma unroll
    for(i32 offset = 0; offset < GPLDA_HASH_STASH_SIZE / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < GPLDA_HASH_STASH_SIZE) {
        this->stash[i] = GPLDA_HASH_EMPTY;
      }
    }
  }

  __device__ __forceinline__ void rebuild() {

  }

  __device__ __forceinline__ u32 get(u32 key) {
    // check table
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
      i32 slot = hash_fn(key, this->a[i], this->b[i], this->size);
      int64_t kv = this->data[slot];
      if(left_32_bits(kv) == key) {
        return right_32_bits(kv);
      }
    }

    // check stash
    i32 slot = hash_fn(key, this->a_stash, this->b_stash, GPLDA_HASH_STASH_SIZE);
    int64_t kv = this->data[slot];
    if(left_32_bits(kv) == key) {
      return right_32_bits(kv);
    }

    // no value: return zero
    return 0;
  }

  __device__ __forceinline__ void set(u32 key, u32 value) {
    u64 kv = ((u64) key << 32) | value;
    i32 a = this->a[0];
    i32 b = this->b[0];
    for(i32 i = 0; i < GPLDA_HASH_MAX_ITERATIONS; ++i) {
      i32 slot = hash_fn(key, a, b, this->size);
      kv = atomicExch(&this->data[slot],kv);
      key = left_32_bits(kv);

      // if slot was empty, exit
      if(kv == GPLDA_HASH_EMPTY) {
        break; // don't return: might need to rebuild on that warp
      }

      // determine which hash function was used, try again
      i32 j = hash_idx(key, slot, this->a, this->b, this->size);
      a = this->a[j+1];
      b = this->b[j+1];
    }

    // if key is still present, try stash
    if(kv != GPLDA_HASH_EMPTY) {
      i32 slot = hash_fn(key, this->a_stash, this->b_stash, GPLDA_HASH_STASH_SIZE);
      kv = atomicExch(&this->stash[slot], kv);
    }

  //  __syncthreads();

    // if key is still present, stash collided, so rebuild table
    if(kv != GPLDA_HASH_EMPTY) {
      rebuild();
    }
  }

  __device__ __forceinline__ void accumulate(u32 key, i32 diff) {
    u32 value = get(key);
  }
};

}
