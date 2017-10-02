#include "test_hashmap.cuh"
#include "../hashmap_robinhood.cuh"
#include "../random.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;
using gplda::f32;
using gplda::i32;
using gplda::u32;
using gplda::u64;

namespace gplda_test {

template<gplda::SynchronizationType sync_type>
__global__ void test_hash_map_init(void* map_storage, u32 total_map_size, u32 initial_size, u32 num_concurrent_elements, u32* map_returned_size, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap<sync_type> m[1];
  m->init(map_storage, total_map_size, initial_size, num_concurrent_elements, rng);
  if(threadIdx.x == 0) {
    map_returned_size[0] = m->size;
  }
}

//  m->a=26; m->b=1; m->c=30; m->d=13;
//
//  // 16
//  m->accumulate2(threadIdx.x < 16 ? 0 : 3, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 6 : 9, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 12 : 15, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 18 : 21, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 24 : 27, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 30 : 33, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 36 : 39, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 42 : 45, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//
//  // 16->32
//  m->accumulate2(threadIdx.x < 16 ? 48 : 51, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 54 : 57, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 60 : 63, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 66 : 69, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 72 : 75, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 78 : 81, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 84 : 87, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 90 : 93, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//
//  // 48
//  m->accumulate2(threadIdx.x < 16 ? 1 : 4, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 7 : 10, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 13 : 16, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 19 : 22, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 25 : 28, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 31 : 34, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 37 : 40, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 43 : 46, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//
//  // 16->48 evict
//  m->accumulate2(threadIdx.x < 16 ? 96 : 99, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 102 : 105, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 108 : 111, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 114 : 117, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 120 : 123, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 126 : 129, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 132 : 135, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//  m->accumulate2(threadIdx.x < 16 ? 138 : 141, 1); if(threadIdx.x == 0) printf("------------------------------------------------------------\n");
//
//  m->debug_print_slot(0, 0, "");
//  m->debug_print_slot(16, 0, "");
//  m->debug_print_slot(32, 0, "");
//  m->debug_print_slot(48, 0, "");
//  m->debug_print_slot(64, 0, "");
//  m->debug_print_slot(80, 0, "");

template<gplda::SynchronizationType sync_type, i32 rebuild>
__global__ void test_hash_map_accumulate2(void* map_storage, u32 total_map_size, u32 num_unique_elements, u32 num_elements, u32 max_size, u32 num_concurrent_elements, u32* out, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap<sync_type> m[1];
  u32 initial_size = rebuild ? num_elements : max_size;
  m->init(map_storage, total_map_size, initial_size, num_concurrent_elements, rng);
  i32 dim = (sync_type == gplda::block) ? blockDim.x / (warpSize / 2) : warpSize / (warpSize / 2);
  i32 half_warp_idx = threadIdx.x / (warpSize / 2);
  i32 half_lane_idx = threadIdx.x % (warpSize / 2);

  // accumulate elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    u32 i = offset * dim + half_warp_idx;
    m->accumulate2(i % num_unique_elements, i < num_elements ? 1 : 0);
  }

  // sync if needed
  if(sync_type == gplda::block) {
    __syncthreads();
  }

  // rebuild if needed
  if(rebuild == true) {
    m->trigger_resize(m->empty_key(), 0);
  }

  // sync if needed
  if(sync_type == gplda::block) {
    __syncthreads();
  }

  // retrieve elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    i32 i = offset * dim + half_warp_idx;
    if(i < num_unique_elements) {
      u32 element = m->get2(i);
      if(half_lane_idx == 0) {
        out[i] = element;
      }
    }
  }
}

void test_hash_map() {
  constexpr u32 max_size = 100; // will round down to 96 for cache alignment
  constexpr u32 num_elements = 90; // large contention to ensure collisions occur
  constexpr u32 num_unique_elements = 9;
  constexpr u32 warpSize = 32;
  constexpr u32 num_concurrent_elements = GPLDA_POLYA_URN_SAMPLE_BLOCKDIM/(warpSize/2);
  constexpr u32 total_map_size = 2*max_size + 3*num_concurrent_elements;
  constexpr u64 empty = (((u64) 0x7f) << 56) | (((u64) 0xfffff) << 36);

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  gplda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  void* map;
  cudaMalloc(&map, total_map_size * sizeof(u64)) >> GPLDA_CHECK;
  u64* map_host = new u64[total_map_size];

  u32* out;
  cudaMalloc(&out, num_elements * sizeof(u32)) >> GPLDA_CHECK;
  u32* out_host = new u32[num_elements];

  // init<warp>
  test_hash_map_init<gplda::warp><<<1,warpSize>>>(map, total_map_size, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, total_map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(out_host[0] == (max_size / GPLDA_HASH_LINE_SIZE) * GPLDA_HASH_LINE_SIZE);
  for(i32 i = 0; i < out_host[0]; ++i) {
    assert(map_host[i] == empty);
    map_host[i] = 0;
  }
  out_host[0] = 0;

  // init<block>
  test_hash_map_init<gplda::block><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, total_map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(out_host[0] == (max_size / GPLDA_HASH_LINE_SIZE) * GPLDA_HASH_LINE_SIZE);
  for(i32 i = 0; i < out_host[0]; ++i) {
    assert(map_host[i] == empty);
    map_host[i] = 0;
  }
  out_host[0] = 0;

  // accumulate2<warp, no_rebuild>
  test_hash_map_accumulate2<gplda::warp, false><<<1,warpSize>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // accumulate2<block, no_rebuild>
  test_hash_map_accumulate2<gplda::block, false><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // accumulate2<warp, rebuild>
  test_hash_map_accumulate2<gplda::warp, true><<<1,warpSize>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // accumulate2<block, rebuild>
  test_hash_map_accumulate2<gplda::block, true><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }




  // cleanup
  cudaFree(out);
  delete[] out_host;
  cudaFree(map);
  delete[] map_host;
  cudaFree(rng);
}


}
