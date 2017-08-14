#pragma once

#include "stdint.h"

namespace gplda {

struct HashMap {
  uint32_t size;
  uint32_t num_elements;
  uint64_t* data;
  uint64_t* stash;
  uint64_t* temp_data;
  uint64_t* temp_stash;
};

__device__ __forceinline__ void hash_map_init(HashMap* map, void* temp, uint32_t size) {
  map->size = size;
  map->num_elements = 0;
  // determine random hash functions
}

__device__ __forceinline__ void hash_map_increment(uint32_t key, HashMap* map) {

}

__device__ __forceinline__ uint32_t hash_map_get(uint32_t key, HashMap* map) {
  return 0;
}

}
