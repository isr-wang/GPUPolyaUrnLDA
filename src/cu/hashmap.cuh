#pragma once

#include "stdint.h"

#define GPLDA_EMPTY_HASH 0xffffffff00000000

namespace gplda {

struct HashMap {
  uint32_t size;
  uint32_t num_elements;
  uint64_t* data;
  uint64_t* stash;
  uint64_t* temp_data;
  uint64_t* temp_stash;
};

__device__ __forceinline__ void hash_map_init(HashMap* map, void* temp, uint32_t size, int32_t dim) {
  // determine constants
  int32_t lane_idx = threadIdx.x % dim;

  // set map parameters and calculate random hash functions
  map->size = size;
  map->num_elements = 0;
  map->data = (uint64_t*) temp;

  // set map to zero
  for(int32_t offset = 0; offset < size / dim + 1; ++offset) {
    // misaligned address?
    map->data[offset * dim + lane_idx] = GPLDA_EMPTY_HASH;
  }
}

__device__ __forceinline__ void hash_map_increment(uint32_t key, HashMap* map) {

}

__device__ __forceinline__ uint32_t hash_map_get(uint32_t key, HashMap* map) {
  return 0;
}

}
