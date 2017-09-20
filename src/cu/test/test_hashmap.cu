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
__global__ void test_hash_map_init(void* map_storage, u32 total_map_size, u32 initial_size, u32 num_concurrent_elements, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap<sync_type> m[1];
  m->init(map_storage, total_map_size, initial_size, num_concurrent_elements, rng);
}

template<gplda::SynchronizationType sync_type, i32 rebuild>
__global__ void test_hash_map_accumulate2(void* map_storage, u32 total_map_size, u32 num_unique_elements, u32 num_elements, u32 max_size, u32 num_concurrent_elements, u32* out, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap<sync_type> m[1];
  u32 initial_size = rebuild ? num_elements : max_size;
  m->init(map_storage, total_map_size, initial_size, num_concurrent_elements, rng);
  i32 dim = (sync_type == gplda::block) ? blockDim.x / (warpSize / 2) : warpSize / (warpSize / 2);
  i32 half_warp_idx = threadIdx.x / (warpSize / 2);

  // accumulate elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    u32 i = offset * dim + half_warp_idx;
    m->accumulate2(i % num_unique_elements, i < num_elements ? 1 : 0);
  }

  // sync if needed
  if(sync_type == gplda::block) {
    __syncthreads();
  }

  // retrieve elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    i32 i = offset * dim + half_warp_idx;
    if(i < num_unique_elements) {
      out[i] = m->get2(i);
    }
  }
}



void test_hash_map() {
  constexpr u32 max_size = 100;
  constexpr u32 num_elements = 90; // large contention to ensure collisions occur
  constexpr u32 num_unique_elements = 9;
  constexpr u32 num_concurrent_elements = GPLDA_POLYA_URN_SAMPLE_BLOCKDIM/16;
  constexpr u32 total_map_size = 2*max_size + 3*num_concurrent_elements;
  constexpr u32 warpSize = 32;

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
  test_hash_map_init<gplda::warp><<<1,warpSize>>>(map, total_map_size, max_size, num_concurrent_elements, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, total_map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < max_size; ++i) {
    assert(map_host[i] == GPLDA_HASH_EMPTY);
    map_host[i] = 0;
  }

  // init<block>
  test_hash_map_init<gplda::block><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, max_size, num_concurrent_elements, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, total_map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < max_size; ++i) {
    assert(map_host[i] == GPLDA_HASH_EMPTY);
    map_host[i] = 0;
  }

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
