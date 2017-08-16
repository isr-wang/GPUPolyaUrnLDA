#pragma once

#include "stdint.h"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#define GPLDA_HASH_EMPTY 0xffffffff00000000

namespace gplda {

struct HashMap {
  uint32_t size;
  uint32_t num_elements;
  uint64_t* data;
  uint64_t* stash;
  uint64_t* temp_data;
  uint64_t* temp_stash;
  uint32_t a[GPLDA_HASH_NUM_FUNCTIONS];
  uint32_t b[GPLDA_HASH_NUM_FUNCTIONS];
  uint32_t a_stash;
  uint32_t b_stash;
};

__device__ __forceinline__ uint32_t left_32_bits(uint64_t x) {
  return (uint32_t) (x >> 32);
}

__device__ __forceinline__ uint32_t right_32_bits(uint64_t x) {
  return (uint32_t) x;
}

__device__ __forceinline__ int32_t hash_fn(int32_t k, int32_t a, int32_t b, int32_t s) {
  return ((a * k + b) % 334214459) % s;
}

__device__ __forceinline__ int32_t hash_idx(uint32_t key, int32_t slot, uint32_t* a, uint32_t* b, uint32_t size) {
  #pragma unroll
  for(int32_t i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
    int32_t possible_slot = hash_fn(key, a[i], b[i], size);
    if(possible_slot == slot) {
      return i;
    }
  }
  return 0;
}

__device__ inline void hash_map_init(HashMap* map, void* temp, uint32_t size, int32_t dim, curandStatePhilox4_32_10_t* rng) {
  // determine constants
  int32_t thread_idx = threadIdx.x % dim;

  // set map parameters and calculate random hash functions
  map->size = size;
  map->num_elements = 0;
  map->data = (uint64_t*) temp;
  map->stash = map->data + size*sizeof(uint64_t);
  map->temp_data = map->stash + GPLDA_HASH_STASH_SIZE*sizeof(uint64_t);
  map->temp_stash = map->temp_data + size*sizeof(uint64_t);
  #pragma unroll
  for(int32_t i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
    map->a[i] = __float2uint_rz(size * curand_uniform(rng));
    map->b[i] = __float2uint_rz(size * curand_uniform(rng));
  }
  map->a_stash = __float2uint_rz(size * curand_uniform(rng));
  map->b_stash = __float2uint_rz(size * curand_uniform(rng));

  // set map to empty
  for(int32_t offset = 0; offset < size / dim + 1; ++offset) {
    int32_t i = offset * dim + thread_idx;
    if(i < size) {
      map->data[i] = GPLDA_HASH_EMPTY;
    }
  }

  // set stash to empty
  #pragma unroll
  for(int32_t offset = 0; offset < GPLDA_HASH_STASH_SIZE / dim + 1; ++offset) {
    int32_t i = offset * dim + thread_idx;
    if(i < GPLDA_HASH_STASH_SIZE) {
      map->stash[i] = GPLDA_HASH_EMPTY;
    }
  }
}

__device__ __forceinline__ uint32_t hash_map_get(uint32_t key, HashMap* map) {
  // check table
  #pragma unroll
  for(int32_t i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
    int32_t slot = hash_fn(key, map->a[i], map->b[i], map->size);
    int64_t kv = map->data[slot];
    if(left_32_bits(kv) == key) {
      return right_32_bits(kv);
    }
  }

  // check stash
  int32_t slot = hash_fn(key, map->a_stash, map->b_stash, GPLDA_HASH_STASH_SIZE);
  int64_t kv = map->data[slot];
  if(left_32_bits(kv) == key) {
    return right_32_bits(kv);
  }

  // no value: return zero
  return 0;
}

__device__ __forceinline__ void hash_map_set(uint32_t key, uint32_t value, HashMap* map) {
  uint64_t kv = ((uint64_t) key << 32) | value;
  int32_t a = map->a[0];
  int32_t b = map->b[0];
  #pragma unroll
  for(int32_t i = 0; i < GPLDA_HASH_MAX_ITERATIONS; ++i) {
    int32_t slot = hash_fn(key, a, b, map->size);
    kv = atomicExch((unsigned long long int*) &map->data[slot],(unsigned long long int) kv);
    key = left_32_bits(kv);

    // if slot was empty, exit
    if(kv == GPLDA_HASH_EMPTY) {
      break; // don't return: might need to rebuild on that warp
    }

    // determine which hash function was used, try again
    int32_t j = hash_idx(key, slot, map->a, map->b, map->size);
    a = map->a[j+1];
    b = map->b[j+1];
  }

  // if key is still present, try stash
  if(kv != GPLDA_HASH_EMPTY) {
    int32_t slot = hash_fn(key, map->a_stash, map->b_stash, GPLDA_HASH_STASH_SIZE);
    kv = atomicExch((unsigned long long int*) &map->stash[slot], (unsigned long long int) kv);
  }

  // if key is still present, stash collided, so rebuild table
  if(kv != GPLDA_HASH_EMPTY) {
    //hash_map_rebuild();
  }
}

__device__ __forceinline__ void hash_map_accumulate(uint32_t key, int32_t diff, HashMap* map) {
  uint32_t value = hash_map_get(key, map);
}
}
