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

__global__ void test_hash_map_init(void* map_storage, u32 total_map_size, u32 initial_size, u32 num_concurrent_elements, u32* map_returned_size, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, total_map_size, initial_size, num_concurrent_elements, rng, blockDim.x);
  __syncthreads();

  if(threadIdx.x == 0) {
    map_returned_size[0] = m->size;
  }
}

__global__ void test_hash_map_insert_print_steps(void* map_storage, curandStatePhilox4_32_10_t* rng) {
  #ifdef GPLDA_HASH_DEBUG
  // initialize
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, 204, 96, 4, rng, blockDim.x);
  m->a=26; m->b=1; m->c=30; m->d=13;
  __syncthreads();

  // 16
  m->insert2(threadIdx.x < 16 ? 0 : 3, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 6 : 9, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 12 : 15, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 18 : 21, 1);if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 24 : 27, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 30 : 33, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 36 : 39, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 42 : 45, 1); if(threadIdx.x == 0) { m->debug_print_slot(16); printf("------------------------------------------------------------\n"); }

  // 16->32
  m->insert2(threadIdx.x < 16 ? 48 : 51, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 54 : 57, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 60 : 63, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 66 : 69, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 72 : 75, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 78 : 81, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 84 : 87, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 90 : 93, 1); if(threadIdx.x == 0) { m->debug_print_slot(32); printf("------------------------------------------------------------\n"); }

  // 48
  m->insert2(threadIdx.x < 16 ? 1 : 4, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 7 : 10, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 13 : 16, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 19 : 22, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 25 : 28, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 31 : 34, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 37 : 40, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 43 : 46, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }

  // 16->48 evict
  m->insert2(threadIdx.x < 16 ? 96 : 99, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 102 : 105, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 108 : 111, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 114 : 117, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 120 : 123, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 126 : 129, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 132 : 135, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }
  m->insert2(threadIdx.x < 16 ? 138 : 141, 1); if(threadIdx.x == 0) { m->debug_print_slot(48); printf("------------------------------------------------------------\n"); }

  if(threadIdx.x == 0) { m->debug_print_slot(0); }
  if(threadIdx.x == 0) { m->debug_print_slot(16); }
  if(threadIdx.x == 0) { m->debug_print_slot(32); }
  if(threadIdx.x == 0) { m->debug_print_slot(48); }
  if(threadIdx.x == 0) { m->debug_print_slot(64); }
  if(threadIdx.x == 0) { m->debug_print_slot(80); }
  #endif
}





__global__ void test_hash_map_insert_phase_1_search(void* map_storage, u32* error, curandStatePhilox4_32_10_t* rng) {
  // initialize
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, 204, 96, 4, rng, blockDim.x);
  m->a=26; m->b=1; m->c=30; m->d=13;
  u64 empty = m->entry(false, false, m->null_pointer(), m->empty_key(), 0);
  __syncthreads();

  // variables
  u32 key;
  i32 half_lane_idx = threadIdx.x % (warpSize/2);
  u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));
  i32 insert_failed;
  i32 slot;
  i32 stride;
  u64* thread_address;
  u64 thread_entry;
  i32 swap_idx;
  i32 swap_type;




  // type 1
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  key = 48;
  insert_failed = false; slot = m->hash_slot(key,m->a,m->b); stride = m->hash_slot(key,m->c,m->d);
  m->insert_phase_1_search(key, half_lane_idx, half_lane_mask, insert_failed, slot, stride, thread_address, thread_entry, swap_idx, swap_type);
  __syncthreads();

  if(slot != 32) {
    error[0] = 1;
  } else if(swap_idx != 0) {
    error[0] = 2;
  } else if(swap_type != 1) {
    error[0] = 3;
  } else if(half_lane_idx == 0 && thread_address != &m->data[32]) {
    error[0] = 4;
  } else if(half_lane_idx == 0 && thread_entry != m->data[32]) {
    error[0] = 5;
  }

  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }




  // type 2
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = empty;
  }
  __syncthreads();

  key = 96;
  insert_failed = false; slot = m->hash_slot(key,m->a,m->b); stride = m->hash_slot(key,m->c,m->d);
  m->insert_phase_1_search(key, half_lane_idx, half_lane_mask, insert_failed, slot, stride, thread_address, thread_entry, swap_idx, swap_type);
  __syncthreads();

  if(slot != 48) {
    error[0] = 6;
  } else if(swap_idx != 0) {
    error[0] = 7;
  } else if(swap_type != 2) {
    error[0] = 8;
  } else if(half_lane_idx == 0 && thread_address != &m->data[48]) {
    error[0] = 9;
  } else if(half_lane_idx == 0 && thread_entry != m->data[48]) {
    error[0] = 10;
  }

  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }




  // type 3
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  key = 96;
  insert_failed = false; slot = m->hash_slot(key,m->a,m->b); stride = m->hash_slot(key,m->c,m->d);
  m->insert_phase_1_search(key, half_lane_idx, half_lane_mask, insert_failed, slot, stride, thread_address, thread_entry, swap_idx, swap_type);
  __syncthreads();

  if(slot != 48) {
    error[0] = 11;
  } else if(swap_idx != 0) {
    error[0] = 12;
  } else if(swap_type != 3) {
    error[0] = 13;
  } else if(half_lane_idx == 0 && thread_address != &m->data[48]) {
    error[0] = 14;
  } else if(half_lane_idx == 0 && thread_entry != m->data[48]) {
    error[0] = 15;
  }

  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }
}


__global__ void test_hash_map_insert_phase_1(void* map_storage, u32* error, curandStatePhilox4_32_10_t* rng) {
  // initialize
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, 204, 96, 4, rng, blockDim.x);
  m->a=26; m->b=1; m->c=30; m->d=13;
  u64 empty = m->entry(false, false, m->null_pointer(), m->empty_key(), 0);
  __syncthreads();

  // variables
  u32 key;
  i32 diff;
  i32 lane_idx = threadIdx.x % warpSize;
  i32 half_lane_idx = threadIdx.x % (warpSize/2);
  u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));
  i32 insert_failed;
  i32 slot;
  i32 stride;




  // type 1
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  key = 48;
  diff = 1;
  insert_failed = false; slot = m->hash_slot(key,m->a,m->b); stride = m->hash_slot(key,m->c,m->d);
  m->insert_phase_1(key, diff, lane_idx, half_lane_idx, half_lane_mask, insert_failed, slot, stride);
  __syncthreads();

  if(m->value(m->data[32]) != 3) {
    error[0] = 1;
  } else if(slot != 32) {
    error[0] = 2;
  }

  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }




  // type 2
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = empty;
  }
  __syncthreads();

  key = 96;
  diff = 1;
  insert_failed = false; slot = m->hash_slot(key,m->a,m->b); stride = m->hash_slot(key,m->c,m->d);
  m->insert_phase_1(key, diff, lane_idx, half_lane_idx, half_lane_mask, insert_failed, slot, stride);
  __syncthreads();

  if(m->data[48] != m->entry(false,false, m->null_pointer(), 96, 2)) {
    error[0] = 3;
  } else if(slot != 48) {
    error[0] = 4;
  }

  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }




  // type 3
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  key = 96;
  diff = 1;
  insert_failed = false; slot = m->hash_slot(key,m->a,m->b); stride = m->hash_slot(key,m->c,m->d);
  m->insert_phase_1(key, diff, lane_idx, half_lane_idx, half_lane_mask, insert_failed, slot, stride);
  __syncthreads();

  if(m->pointer(m->data[48]) == m->null_pointer()){
    error[0] = 5;
  } else if(m->ring_buffer[m->pointer(m->data[48])] != m->entry(false,false, m->null_pointer(), 96, 2)) {
    error[0] = 6;
  } else if(slot != 48) {
    error[0] = 7;
  }

  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }
}


__global__ void test_hash_map_insert_phase_2_determine_index(void* map_storage, u32* error, curandStatePhilox4_32_10_t* rng) {
  // variables
  i32 half_lane_idx = threadIdx.x % (warpSize/2);
  u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));
  i32 slot;
  u64 half_warp_entry;
  i32 half_warp_entry_idx;

  // initialize
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, 204, 96, 4, rng, blockDim.x);
  m->a=26; m->b=1; m->c=30; m->d=13;
  __syncthreads();

  // pointer, no relocation bit
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 48, 2);
    m->data[20] = m->with_pointer(ptr, m->data[20]);
  }
  __syncthreads();
  slot = 16;
  m->insert_phase_2_determine_index(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_entry_idx);
  __syncthreads();

  if(half_warp_entry_idx != 4) {
    error[0] = 1;
  } else if(half_warp_entry != m->data[20]) {
    error[0] = 2;
  }

  // no pointer, relocation bit
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    m->data[20] = m->with_relocate(true, m->data[20]);
  }
  __syncthreads();
  slot = 16;
  m->insert_phase_2_determine_index(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_entry_idx);
  __syncthreads();

  if(half_warp_entry_idx != 4) {
    error[0] = 3;
  } else if(half_warp_entry != m->data[20]) {
    error[0] = 4;
  }

  // pointer and relocation bit
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 48, 2);
    m->data[20] = m->with_pointer(ptr, m->data[20]);

    i32 ptr2 = m->ring_buffer_pop();
    m->ring_buffer[ptr2] = m->entry(true,false, m->null_pointer(), 51, 2);
    m->data[21] = m->with_relocate(true,m->with_pointer(ptr, m->data[21]));
  }
  __syncthreads();
  slot = 16;
  m->insert_phase_2_determine_index(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_entry_idx);
  __syncthreads();

  if(half_warp_entry_idx != 5) {
    error[0] = 5;
  } else if(half_warp_entry != m->data[21]) {
    error[0] = 6;
  }
}





__global__ void test_hash_map_insert_phase_2_determine_stage_search(void* map_storage, u32* error, curandStatePhilox4_32_10_t* rng) {
  // variables
  i32 half_lane_idx = threadIdx.x % (warpSize/2);
  u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));
  i32 slot;
  u64 half_warp_entry;
  u64 half_warp_temp;
  i32 half_warp_temp_idx;
  i32 stage;
  u64 half_warp_link_entry;

  // initialize
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, 204, 96, 4, rng, blockDim.x);
  m->a=26; m->b=1; m->c=30; m->d=13;
  __syncthreads();
  half_warp_entry = m->entry(false,false, m->null_pointer(), 0, 1);
  half_warp_link_entry = m->entry(false,false, m->null_pointer(), 96, 2);
  slot = 16;
  u64 empty = m->entry(false, false, m->null_pointer(), m->empty_key(), 0);


  // stage 2: entry not present, will go in queue
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));
  }
  __syncthreads();
  m->insert_phase_2_determine_stage_search(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage, half_warp_link_entry);
  __syncthreads();

  if(stage != 2) {
    error[0] = 1;
  } else if(half_warp_temp != m->data[48]) {
    error[0] = 2;
  } else if(half_warp_temp_idx != 48) {
    error[0] = 3;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }



  // stage 2: entry not present, will go in empty slot
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 48, 2);
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));
  }
  __syncthreads();
  m->insert_phase_2_determine_stage_search(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage, half_warp_link_entry);
  __syncthreads();

  if(stage != 2) {
    error[0] = 4;
  } else if(half_warp_temp != m->entry(false, false, m->null_pointer(), m->empty_key(), 0)) {
    error[0] = 5;
  } else if(half_warp_temp_idx != 32/*empty slot's index*/) {
    error[0] = 6;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }



  // stage 3: entry in queue
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));

    i32 ptr2 = m->ring_buffer_pop();
    m->ring_buffer[ptr2] = m->with_pointer(m->null_pointer(), m->with_relocate(false, m->data[16]));
    m->data[48] = m->with_pointer(ptr2, m->data[48]);
  }
  __syncthreads();
  m->insert_phase_2_determine_stage_search(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage, half_warp_link_entry);
  __syncthreads();

  if(stage != 3) {
    error[0] = 7;
  } else if(half_warp_temp != m->ring_buffer[m->pointer(m->data[16])]) {
    error[0] = 8;
  } else if(half_warp_temp_idx != -1) {
    error[0] = 9;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }



  // stage 3: entry in table
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));
    m->data[32] = m->with_pointer(m->null_pointer(), m->with_relocate(false, m->data[16]));
  }
  __syncthreads();
  m->insert_phase_2_determine_stage_search(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage, half_warp_link_entry);
  __syncthreads();

  if(stage != 3) {
    error[0] = 10;
  } else if(half_warp_temp != m->ring_buffer[m->pointer(m->data[16])]) {
    error[0] = 11;
  } else if(half_warp_temp_idx != -1) {
    error[0] = 12;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }



}






__global__ void test_hash_map_insert_phase_2_determine_stage(void* map_storage, u32* error, curandStatePhilox4_32_10_t* rng) {
  // initialize
  __shared__ gplda::HashMap m[1];
  m->init(map_storage, 204, 96, 4, rng, blockDim.x);
  m->a=26; m->b=1; m->c=30; m->d=13;
  u64 empty = m->entry(false, false, m->null_pointer(), m->empty_key(), 0);
  __syncthreads();

  // variables
  i32 half_lane_idx = threadIdx.x % (warpSize/2);
  u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));
  u64 half_warp_entry = m->entry(false,false, m->null_pointer(), 0, 1);
  i32 slot = 16;
  u64 half_warp_temp;
  i32 half_warp_temp_idx;
  i32 stage;






  // stage 1
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->data[16]);
  }
  __syncthreads();

  half_warp_entry = m->data[16];
  m->insert_phase_2_determine_stage(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage);
  __syncthreads();

  if(stage != 1) {
    error[0] = 1;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }






  // stage 2
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));
  }
  __syncthreads();

  half_warp_entry = m->data[16];
  m->insert_phase_2_determine_stage(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage);
  __syncthreads();

  if(stage != 2) {
    error[0] = 2;
  } else if(half_warp_temp != m->data[48]) {
    error[0] = 3;
  } else if(half_warp_temp_idx != 48) {
    error[0] = 4;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }




  // stage 3
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
    m->data[32+threadIdx.x] = m->entry(false,false, m->null_pointer(), 48 + 3*threadIdx.x, 1);
    m->data[48+threadIdx.x] = m->entry(false,false, m->null_pointer(), 1 + 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(false,false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));

    i32 ptr2 = m->ring_buffer_pop();
    m->ring_buffer[ptr2] = m->with_pointer(m->null_pointer(), m->with_relocate(false, m->data[16]));
    m->data[48] = m->with_pointer(ptr2, m->data[48]);
  }
  __syncthreads();

  half_warp_entry = m->data[16];
  m->insert_phase_2_determine_stage(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage);
  __syncthreads();

  if(stage != 3) {
    error[0] = 5;
  } else if(half_warp_temp != m->ring_buffer[m->pointer(m->data[16])]) {
    error[0] = 6;
  } else if(half_warp_temp_idx != -1) {
    error[0] = 7;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }





  // stage 4
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  if(threadIdx.x == 0) {
    i32 ptr = m->ring_buffer_pop();
    m->ring_buffer[ptr] = m->entry(true, false, m->null_pointer(), 96, 2); // will evict slot 48
    m->data[16] = m->with_pointer(ptr, m->with_relocate(true,m->data[16]));
  }
  __syncthreads();

  half_warp_entry = m->data[16];
  m->insert_phase_2_determine_stage(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage);
  __syncthreads();

  if(stage != 4) {
    error[0] = 8;
  } else if(half_warp_temp != m->ring_buffer[m->pointer(m->data[16])]) {
    error[0] = 9;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }





  // stage 5
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = m->entry(false,false, m->null_pointer(), 3*threadIdx.x, 1);
  }
  __syncthreads();

  half_warp_entry = m->data[16];
  m->insert_phase_2_determine_stage(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_temp, half_warp_temp_idx, stage);
  __syncthreads();

  if(stage != 5) {
    error[0] = 10;
  }

  stage = -1;
  half_warp_temp = 0;
  half_warp_temp_idx = -1;
  if(threadIdx.x < 16) {
    m->data[16+threadIdx.x] = empty;
    m->data[32+threadIdx.x] = empty;
    m->data[48+threadIdx.x] = empty;
  }
}






__global__ void test_hash_map_insert2(void* map_storage, u32 total_map_size, u32 num_unique_elements, u32 num_elements, u32 max_size, u32 num_concurrent_elements, u32* out, curandStatePhilox4_32_10_t* rng, i32 rebuild) {
  __shared__ gplda::HashMap m[1];
  u32 initial_size = rebuild ? num_elements : max_size;
  m->init(map_storage, total_map_size, initial_size, num_concurrent_elements, rng, blockDim.x);
  i32 dim = blockDim.x / (warpSize / 2);
  i32 half_warp_idx = threadIdx.x / (warpSize / 2);
  i32 half_lane_idx = threadIdx.x % (warpSize / 2);
  __syncthreads();

  // accumulate elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    u32 i = offset * dim + half_warp_idx;
    m->insert2(i % num_unique_elements, i < num_elements ? 1 : 0);
  }

  // sync if needed
  __syncthreads();

  // rebuild if needed
  if(rebuild == true) {
    m->resize_table();
  }

  // sync if needed
  __syncthreads();

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



void test_hash_map_phase_1() {
  constexpr u32 max_size = 96;
  constexpr u32 warpSize = 32;
  constexpr u32 num_concurrent_elements = 4;
  constexpr u32 total_map_size = 2*max_size + 3*num_concurrent_elements;

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  gplda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  void* map;
  cudaMalloc(&map, total_map_size * sizeof(u64)) >> GPLDA_CHECK;

  u32* out;
  cudaMalloc(&out, sizeof(u32)) >> GPLDA_CHECK;
  u32 out_host = 0;

  // phase 1 search
  test_hash_map_insert_phase_1_search<<<1,warpSize>>>(map, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  assert(out_host == 0);

  // phase 1
  test_hash_map_insert_phase_1<<<1,warpSize>>>(map, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  assert(out_host == 0);

  // cleanup
  cudaFree(out);
  cudaFree(map);
  cudaFree(rng);
}




void test_hash_map_phase_2() {
  constexpr u32 max_size = 96;
  constexpr u32 warpSize = 32;
  constexpr u32 num_concurrent_elements = 4;
  constexpr u32 total_map_size = 2*max_size + 3*num_concurrent_elements;

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  gplda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  void* map;
  cudaMalloc(&map, total_map_size * sizeof(u64)) >> GPLDA_CHECK;

  u32* out;
  cudaMalloc(&out, sizeof(u32)) >> GPLDA_CHECK;
  u32 out_host = 0;

  // phase 2 determine index
  test_hash_map_insert_phase_2_determine_index<<<1,warpSize>>>(map, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  assert(out_host == 0);

  // phase 2 determine stage search
  test_hash_map_insert_phase_2_determine_stage_search<<<1,warpSize>>>(map, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  assert(out_host == 0);

  // phase 2 determine stage
  test_hash_map_insert_phase_2_determine_stage<<<1,warpSize>>>(map, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  assert(out_host == 0);

  // test_hash_map_insert_print_steps<<<1,warpSize>>>(map, rng);
  // cudaDeviceSynchronize() >> GPLDA_CHECK;

  // cleanup
  cudaFree(out);
  cudaFree(map);
  cudaFree(rng);
}





void test_hash_map() {
  constexpr u32 max_size = 100; // will round down to 96 for cache alignment
  constexpr u32 num_elements = 90; // large contention to ensure collisions occur
  constexpr u32 num_unique_elements = 9;
  constexpr u32 warpSize = 32;
  constexpr u32 num_concurrent_elements = GPLDA_POLYA_URN_SAMPLE_BLOCKDIM/(warpSize/2);
  constexpr u32 total_map_size = 2*max_size + 3*num_concurrent_elements;
  constexpr u64 empty = (((u64) 0x7f) << 55) | (((u64) 0xfffff) << 35);

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
  test_hash_map_init<<<1,warpSize>>>(map, total_map_size, max_size, num_concurrent_elements, out, rng);
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
  test_hash_map_init<<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, max_size, num_concurrent_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, total_map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(out_host[0] == (max_size / GPLDA_HASH_LINE_SIZE) * GPLDA_HASH_LINE_SIZE);
  for(i32 i = 0; i < out_host[0]; ++i) {
    assert(map_host[i] == empty);
    map_host[i] = 0;
  }
  out_host[0] = 0;

  // insert2: warp, no rebuild
  test_hash_map_insert2<<<1,warpSize>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng, false);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: block, no rebuild
  test_hash_map_insert2<<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng, false);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // // accumulate2<warp, rebuild>
  // test_hash_map_accumulate2<gplda::warp, true><<<1,warpSize>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng);
  // cudaDeviceSynchronize() >> GPLDA_CHECK;
  //
  // cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  //
  // for(i32 i = 0; i < num_unique_elements; ++i) {
  //   assert(out_host[i] == num_elements / num_unique_elements);
  //   out_host[i] = 0;
  // }
  //
  // // accumulate2<block, rebuild>
  // test_hash_map_accumulate2<gplda::block, true><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, total_map_size, num_unique_elements, num_elements, max_size, num_concurrent_elements, out, rng);
  // cudaDeviceSynchronize() >> GPLDA_CHECK;
  //
  // cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  //
  // for(i32 i = 0; i < num_unique_elements; ++i) {
  //   assert(out_host[i] == num_elements / num_unique_elements);
  //   out_host[i] = 0;
  // }




  // cleanup
  cudaFree(out);
  delete[] out_host;
  cudaFree(map);
  delete[] map_host;
  cudaFree(rng);
}


}
