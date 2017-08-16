#pragma once

#include "stdint.h"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#define GPLDA_EMPTY_HASH 0xffffffff00000000

namespace gplda {

struct HashMap {
  uint32_t size;
  uint32_t num_elements;
  uint64_t* data;
  uint64_t* stash;
  uint64_t* temp_data;
  uint64_t* temp_stash;
  uint64_t a[GPLDA_HASH_NUM_FUNCTIONS];
  uint64_t b[GPLDA_HASH_NUM_FUNCTIONS];
};

__device__ __forceinline__ int32_t hash_map_hash_function(int32_t k, int32_t a, int32_t b, int32_t s) {
  return (a * k + b) % 334214459 % s;
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
  for(int32_t i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
    map->a[i] = __uint2float_rz(size * curand_uniform(rng));
    map->b[i] = __uint2float_rz(size * curand_uniform(rng));
  }

  // set map to empty
  for(int32_t offset = 0; offset < size / dim + 1; ++offset) {
    int32_t i = offset * dim + thread_idx;
    if(i < size) {
      map->data[i] = GPLDA_EMPTY_HASH;
    }
  }

  // set stash to empty
  for(int32_t offset = 0; offset < GPLDA_HASH_STASH_SIZE / dim + 1; ++offset) {
    int32_t i = offset * dim + thread_idx;
    if(i < GPLDA_HASH_STASH_SIZE) {
      map->stash[i] = GPLDA_EMPTY_HASH;
    }
  }
}

__device__ __forceinline__ uint32_t hash_map_get(uint32_t key, HashMap* map) {
  return 0;
}

__device__ __forceinline__ void hash_map_set(uint32_t key, HashMap* map) {

}

__device__ __forceinline__ void hash_map_increment(uint32_t key, HashMap* map) {
  uint32_t value = hash_map_get(key, map);
}

__device__ __forceinline__ void hash_map_decrement(uint32_t key, HashMap* map) {
  uint32_t value = hash_map_get(key, map);
}
}
