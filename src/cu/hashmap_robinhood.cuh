#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#define GPLDA_HASH_LINE_SIZE 16
#define GPLDA_HASH_MAX_NUM_LINES 4
#define GPLDA_HASH_GROWTH_RATE 1.2f
#define GPLDA_HASH_DEBUG 1

#ifdef GPLDA_HASH_DEBUG
#include <cstdio>
#endif

namespace gplda {

template<SynchronizationType sync_type>
struct HashMap {
  u32 size;
  u32 max_size;
  u64* data;
  u64* temp_data;
  u32 a;
  u32 b;
  u32 c;
  u32 d;
  u32 rebuild_size;
  u32 rebuild_idx;
  u32 rebuild_check;
  u32 rebuild_a;
  u32 rebuild_b;
  u32 rebuild_c;
  u32 rebuild_d;
  curandStatePhilox4_32_10_t* rng;
  u32 ring_buffer_start;
  u32 ring_buffer_read_end;
  u32 ring_buffer_write_end;
  u32 ring_buffer_size;
  u32* ring_buffer_queue;
  u64* ring_buffer;




  __device__ __forceinline__ void sync() {
    if(sync_type == block) {
      __syncthreads();
    }
  }








  __device__ __forceinline__ u64 bfe_b64(u64 source, u32 bit_start, u32 num_bits) {
    u64 bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"(source), "r"(bit_start), "r"(num_bits));
    return bits;
  }

  __device__ __forceinline__ void bfi_b64(u64 &ret, u64 x, u64 y, u32 bit_start, u32 num_bits) {
    asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(ret) : "l"(y), "l"(x), "r"(bit_start), "r"(num_bits));
  }

  __device__ __forceinline__ u64 entry(u32 relocate, u32 pointer, u32 key, u64 value) {
    u64 ret;
    bfi_b64(ret, value, key, 36, 20);
    bfi_b64(ret, ret, pointer, 56, 7);
    bfi_b64(ret, ret, relocate, 63, 1);
    return ret;
  }

  __device__ __forceinline__ u64 with_relocate(u32 relocate, u64 entry) {
    bfi_b64(entry, entry, relocate, 63, 1);
    return entry;
  }

  __device__ __forceinline__ u64 with_pointer(u32 pointer, u64 entry) {
    bfi_b64(entry, entry, pointer, 56, 7);
    return entry;
  }

  __device__ __forceinline__ u64 with_key(u32 key, u64 entry) {
    bfi_b64(entry, entry, key, 36, 20);
    return entry;
  }

  __device__ __forceinline__ u64 with_value(u64 value, u64 entry) {
    bfi_b64(entry, entry, value, 0, 36);
    return entry;
  }

  __device__ __forceinline__ u32 relocate(u64 entry) {
    return bfe_b64(entry, 63, 1);
  }

  __device__ __forceinline__ u32 pointer(u64 entry) {
    return bfe_b64(entry, 56, 7);
  }

  __device__ __forceinline__ u32 key(u64 entry) {
    return bfe_b64(entry, 36, 20);
  }

  __device__ __forceinline__ u64 value(u64 entry) {
    return bfe_b64(entry, 0, 36);
  }

  __device__ __forceinline__ static constexpr u32 null_pointer() {
    return 0x7f;
  }

  __device__ __forceinline__ static constexpr u32 resize_pointer() {
    return 0x7e;
  }

  __device__ __forceinline__ static constexpr u32 empty_key() {
    return 0xfffff;
  }

  __device__ __forceinline__ static constexpr u64 empty_resize() {
    return (((u64) resize_pointer()) << 56) | (((u64) empty_key()) << 36);
  }

  __device__ __forceinline__ static constexpr u64 empty() {
    return (((u64) null_pointer()) << 56) | (((u64) empty_key()) << 36);
  }






  #ifdef GPLDA_HASH_DEBUG
  __device__ inline void debug_print_slot(u32 slot, u32 thread_idx, const char* title) {
    if(threadIdx.x == thread_idx) {
      printf(title);
      printf("\n");
      printf("hl:s\tr\tp\tk\tv\tis:st:d\n");
      for(u32 s = slot; s < slot + warpSize/2; ++s) {
        u64 entry = data[s % size];
        printf("%d:%d\t%d\t%d\t%d\t%d\t", s % 16, s % size, relocate(entry), pointer(entry), key(entry), value(entry));
        if(entry != empty()) printf("%d:%d:%d", hash_slot(key(entry),a,b), hash_slot(key(entry),c,d), key_distance(key(entry), slot));
        while(pointer(entry) != null_pointer()) {
          i32 buffer_idx = pointer(entry);
          entry = ring_buffer[buffer_idx];
          printf("\t-------->\t%d:%d\t%d\t%d\t%d\t%d\t", s % 16, buffer_idx, relocate(entry), pointer(entry), key(entry), value(entry));
          if(entry != empty()) printf("%d:%d:%d", hash_slot(key(entry),a,b), hash_slot(key(entry),c,d), key_distance(key(entry), slot));
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  #endif






  __device__ inline void ring_buffer_push(u32 element) {
    // determine constants
    u32 lane_idx = threadIdx.x % warpSize;

    // divergent threads may enter, so determine which thread will write
    u32 active_threads = __ballot(true);
    u32 leader = __ffs(active_threads) - 1;
    u32 warp_num_threads = __popc(active_threads);
    u32 offset = __popc((~(0xffffffff << lane_idx)) & active_threads);

    // increment queue size once per warp and broadcast to all lanes
    u32 warp_start;
    if(lane_idx == leader) {
      warp_start = atomicAdd(&ring_buffer_write_end, warp_num_threads);
    }
    warp_start = __shfl(warp_start, leader);

    // write elements to queue
    ring_buffer_queue[(warp_start + offset) % size] = element;

    // increment number of elements that may be read from queue
    if(lane_idx == leader) {
      do {} while(atomicCAS(&ring_buffer_read_end, warp_start, warp_start + warp_num_threads) != warp_start);
    }
  }

  __device__ inline u32 ring_buffer_pop() {
    // determine constants
    u32 lane_idx = threadIdx.x % warpSize;

    // divergent threads may enter, so determine which thread will write
    u32 active_threads = __ballot(true);
    u32 leader = __ffs(active_threads) - 1;
    u32 warp_num_threads = __popc(active_threads);
    u32 offset = __popc((~(0xffffffff << lane_idx)) & active_threads);

    // read index from queue and broadcast
    u32 warp_start;
    if(lane_idx == leader) {
      warp_start = atomicAdd(&ring_buffer_start, warp_num_threads);
    }
    warp_start = __shfl(warp_start, leader);

    // return index of buffer location
    return warp_start + offset;
  }

  __device__ inline void ring_buffer_init(u64* b, u32* q, u32 s) {
    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;

    if(thread_idx == 0) {
      ring_buffer_start = 0;
      ring_buffer_read_end = 0;
      ring_buffer_write_end = 0;
      ring_buffer_size = s;
      ring_buffer_queue = q;
      ring_buffer = b;
    }

    // ensure parameter writes are visible to all threads
    sync();

    // set buffer to empty and queue to full
    for(i32 offset = 0; offset < ring_buffer_size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < ring_buffer_size) {
        ring_buffer_queue[i] = i;
        ring_buffer[i] = empty();
      }
    }

    // ensure queue writes are visible to all threads
    sync();
  }






  __device__ __forceinline__ i32 hash_slot(u32 key, i32 x, i32 y) {
    return (((x * key + y) % 334214459) % (size / GPLDA_HASH_LINE_SIZE)) * GPLDA_HASH_LINE_SIZE;
  }

  __device__ __forceinline__ i32 key_distance(u32 key, u32 slot) {
    u32 initial_slot = hash_slot(key,a,b);
    u32 stride = hash_slot(key,c,d);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES - 1; ++i) {
      if((initial_slot + i*stride) % size == slot) {
        return i;
      }
    }
    return GPLDA_HASH_MAX_NUM_LINES;
  }





  __device__ inline void init(void* in_data, u32 in_data_size, u32 initial_size, u32 num_concurrent_elements, curandStatePhilox4_32_10_t* in_rng) {
    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;

    // set map parameters and calculate random hash functions
    if(thread_idx == 0) {
      // round down to ensure cache alignment
      max_size = (((in_data_size - 3*num_concurrent_elements)/2) / GPLDA_HASH_LINE_SIZE) * GPLDA_HASH_LINE_SIZE;
      size = min((initial_size / GPLDA_HASH_LINE_SIZE + 1) * GPLDA_HASH_LINE_SIZE, max_size);

      // perform pointer arithmetic
      data = (u64*) in_data;
      temp_data = data + max_size; // no sizeof for typed pointer arithmetic

      rebuild_size = 0;
      rng = in_rng; // make sure this->rng is set before use
      float4 r = curand_uniform4(rng);
      a = __float2uint_rz(size * r.w);
      b = __float2uint_rz(size * r.x);
      c = __float2uint_rz(size * r.y);
      d = __float2uint_rz(size * r.z);
    }

    // synchronize to ensure shared memory writes are visible
    sync();

    // set map to empty
    for(i32 offset = 0; offset < size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        data[i] = empty();
      }
    }

    ring_buffer_init(temp_data + max_size, (u32*) (temp_data + max_size + 2*num_concurrent_elements), 2*num_concurrent_elements);

    // synchronize to ensure initialization is complete
    sync();
  }




  __device__ inline void trigger_resize(u32 key, u32 diff) {
    if(threadIdx.x % warpSize == 0) {
      // generate new hash functions and set the new size
      float4 r = curand_uniform4(rng);
      atomicCAS(&rebuild_a, a, __float2uint_rz(size * r.w));
      atomicCAS(&rebuild_b, b, __float2uint_rz(size * r.x));
      atomicCAS(&rebuild_c, c, __float2uint_rz(size * r.y));
      atomicCAS(&rebuild_d, d, __float2uint_rz(size * r.z));
      atomicCAS(&rebuild_size, 0, umin(max_size, __float2uint_rz(size * GPLDA_HASH_GROWTH_RATE) + warpSize));
    }

    // place keys that collided first
    attempt_insertion(key,diff);

    // resolve remaining keys
    join_resize();
  }

  __device__ inline void join_resize() {
    // if resize is not in progress, return immediately
    if(__shfl(rebuild_size, 0) == 0) {
      return;
    }

    // compute constants
    i32 lane_idx = threadIdx.x % warpSize;
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);

    // move entries
    resize_move(lane_idx, half_lane_idx);

    // clear any unfinished entries
    resize_clear(lane_idx, half_lane_idx);

    // if everything has been inserted, swap pointers and complete resize
    if(lane_idx == 0 && rebuild_check + 1 >= rebuild_size) {
      // TODO: last warp sets old memory to empty
    }
  }






  __device__ inline void resize_move(i32& lane_idx, i32& half_lane_idx) {
    // iterate over map and place remaining keys
    u32 idx = __shfl(rebuild_idx, 0);
    while(idx < rebuild_size) {
      // increment index
      if(lane_idx == 0) {
        idx = atomicAdd(&rebuild_idx, 2);
      }
      idx = __shfl(idx, 0);
      if(lane_idx >= warpSize/2) {
        idx += 1;
      }

      // if index exceeds table size, exit
      if(idx < size) {
        // repeat until pointers have been exhausted
        i32 repeat;
        do {
          // clear repeat value on all threads
          repeat = false;

          // get entry
          u64 thread_entry;
          u64 thread_previous_entry;
          u64* address;
          u64* previous_address;
          i32 address_buffer = null_pointer();
          if(half_lane_idx == 0) {
            // follow pointers
            address = &data[idx];
            thread_entry = *address;
            while(pointer(thread_entry) != null_pointer()) {
              address_buffer = pointer(thread_entry);
              previous_address = address;
              thread_previous_entry = thread_entry;
              address = &ring_buffer[address_buffer];
              thread_entry = *address;
            };

            // swap in new element
            u64 old = atomicCAS(address, thread_entry, with_pointer(resize_pointer(), thread_entry));

            // don't insert anything if swap failed
            if(old != thread_entry) {
              thread_entry = empty();
            }
          }

          // broadcast thread_entry to entire half warp
          thread_entry = __shfl(thread_entry, 0, warpSize/2);

          // perform insertion
          attempt_insertion(key(thread_entry), value(thread_entry));

          // clear element
          if(half_lane_idx == 0 && thread_entry != empty()) {
            // determine whether element was in ring buffer or in table
            if(address_buffer != null_pointer()) {
              // if element was in ring buffer, unlink it, and then return it
              u64 old = atomicCAS(previous_address, thread_previous_entry, with_pointer(null_pointer(), thread_previous_entry));

              // return to ring buffer, as long as unlinking succeeded, else another thread already did that
              if(old == thread_previous_entry) {
                ring_buffer_push(address_buffer);
              }

              // ensure we don't stop to the next slot in the table
              repeat = true;
            } else {
              // no need to unlink anything because entry was in the table: set it to empty
              atomicCAS(address, thread_entry, empty_resize()); // no need to check for failure: only possible because another thread did it first
            }

          }
        } while(repeat != 0);
      }
    };
  }





  __device__ inline void resize_clear(i32& lane_idx, i32& half_lane_idx) {
    // iterate over map and ensure every key has been cleared
    u32 idx = __shfl(rebuild_check, 0);
    while(idx < rebuild_size) {
      // clear any leftover keys
      u64 thread_entry = idx+lane_idx < size ? data[idx+lane_idx] : empty();
      u32 warp_unfinished_entries = __ballot(thread_entry != empty());

      // TODO: if there are unfinished entries, place those

      // finally, update the index to indicate check is complete
      if(warp_unfinished_entries == 0 && lane_idx == 0) {
        atomicCAS(&rebuild_check, idx, idx+32);
      }
    }
  }





  __device__ inline u32 get2(u32 half_warp_key) {
    // shuffle key to entire half-warp
    half_warp_key = __shfl(half_warp_key, 0, warpSize/2);
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));

    // check table
    i32 initial_slot = hash_slot(half_warp_key,a,b);
    i32 stride = hash_slot(half_warp_key,c,d);
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot and retrieve entry
      i32 slot = (initial_slot + i * stride) % size;
      u64 entry = data[slot + half_lane_idx];

      // check if we found the key, following pointers if necessary
      u32 key_found;
      u32 key_pointer;
      do {
        // check if we found key, return its value if so
        key_found = __ballot(key(entry) == half_warp_key) & half_lane_mask;
        if(key_found != 0) {
          return __shfl(value(entry), __ffs(key_found) - 1, warpSize/2);
        }

        // check if we found pointer, get its entry if so
        key_pointer = __ballot(pointer(entry) != null_pointer()) & half_lane_mask;
        if(key_pointer != 0) {
          entry = ring_buffer[pointer(entry)];
        }
      } while(key_pointer != 0);
      // TODO: check if we read back the same thing in the first slot

      // check if Robin Hood guarantee indicates no key is present
      u32 no_key = __ballot(entry == empty() || key_distance(key(entry), slot) < i) & half_lane_mask;
      if(no_key != 0) {
        return 0;
      }
    }

    // ran out of possible slots: key not present
    return 0;
  }






  __device__ inline void insert_phase_1(u32& half_warp_key, i32& diff, i32& lane_idx, i32& half_lane_idx, u32& half_lane_mask, u64& half_warp_entry, i32& insert_failed, i32& slot, i32& stride) {
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot
      i32 insert_slot = (slot + i*stride) % size;

      // try to insert, retrying if race condition indicates it is necessary
      u32 retry;
      u32 success;
      do {
        // retrieve entry for current half lane, set constants
        u64* thread_address = &data[insert_slot + half_lane_idx];
        u64 thread_table_entry = *thread_address;
        retry = 0;
        success = 0;

        // TODO: don't overwrite relocation bit on linked entry: instead, move it first

        // if there are pointers, follow them to determine distance
        u32 thread_found_key;
        u32 thread_found_empty;
        u32 thread_no_key;
        u32 thread_found_pointer;
        u32 half_warp_found_key;
        u32 half_warp_found_empty;
        u32 half_warp_no_key;
        u32 half_warp_found_pointer;
        do {
          // determine whether we found the key, an empty slot, or no key is present
          thread_found_key = key(thread_table_entry) == half_warp_key;
          thread_found_empty = thread_table_entry == empty();
          thread_no_key = key_distance(key(thread_table_entry), insert_slot) < i;
          thread_found_pointer = pointer(thread_table_entry) != null_pointer();

          // determine which thread should write
          half_warp_found_key = __ballot(thread_found_key) & half_lane_mask;
          half_warp_found_empty = __ballot(thread_found_empty) & half_lane_mask;
          half_warp_no_key = __ballot(thread_no_key) & half_lane_mask;
          half_warp_found_pointer = __ballot(thread_found_pointer) & half_lane_mask;

          if(thread_found_pointer == true) {
            thread_address = &ring_buffer[pointer(thread_table_entry)];
            thread_table_entry = *thread_address;
          }
        } while (half_warp_found_key == 0 && half_warp_found_pointer != 0);

        u32 half_warp_write;
        if(half_warp_found_key != 0) {
          half_warp_write = half_warp_found_key;
        } else if(half_warp_found_empty != 0) {
          half_warp_write = half_warp_found_empty;
        } else if(half_warp_no_key != 0) {
          half_warp_write = half_warp_no_key;
        } else {
          half_warp_write = 0;
        }
        u32 lane_write_idx = __ffs(half_warp_write) - 1; // __ffs uses 1-based indexing

        u64 half_warp_write_entry;
        if(half_warp_write != 0 && lane_idx == lane_write_idx) {
          // prepare new entry for table
          u32 buffer_idx = null_pointer();

          // determine what kind of new entry we have
          if(thread_found_key == true) {
            // key found: accumulate value
            u64 new_value = max((u64) 0, ((u64) value(thread_table_entry)) + diff);
            half_warp_write_entry = with_value(new_value, half_warp_entry);
          } else if(thread_found_empty == true) {
            // empty slot found: insert entry
            half_warp_write_entry = half_warp_entry;
          } else if(thread_no_key == true) {
            // Robin Hood guarantee indicates no key present: insert into eviction queue
            buffer_idx = ring_buffer_pop();
            ring_buffer[buffer_idx] = half_warp_entry;

            // prepare new entry
            half_warp_write_entry = with_pointer(buffer_idx, thread_table_entry);
          }

          // swap new and old entry
          u64 old_entry = atomicCAS(thread_address, thread_table_entry, half_warp_write_entry);

          // make sure retrieved entry matches what was expected, so we know that CAS succeeded
          if(old_entry != thread_table_entry) {
            // set retry indicator
            retry = true;

            // clear buffer, if it was requested
            if(buffer_idx != null_pointer()) {
              ring_buffer[buffer_idx] = empty();
              ring_buffer_push(buffer_idx);
            }
          } else {
            success = true;
          }
        }

        // ensure entire halfwarp knows whether write succeeded
        success = __ballot(success) & half_lane_mask;

        // ensure retry, if necessary, is performed on entire half warp
        retry = __ballot(retry) & half_lane_mask;
      } while(retry != 0);

      // if half warp successfully performed a write, exit the loop
      if(success != 0) {
        slot = insert_slot;
        break;
      } else if(i == GPLDA_HASH_MAX_NUM_LINES - 1) {
        // insertion failed, get ready to return false
        insert_failed = 1;
      }
    }

  }




  __device__ inline void insert_phase_2_determine_index(i32& half_lane_idx, u32& half_lane_mask, i32& slot, u64& half_warp_entry, i32& half_warp_entry_idx) {
    // load entry from table
    u64 thread_table_entry = data[slot + half_lane_idx];
    u32 half_warp_relocation = __ballot(relocate(thread_table_entry) != 0) & half_lane_mask;
    u32 half_warp_pointer = __ballot(pointer(thread_table_entry) != null_pointer()) & half_lane_mask;

    if(half_warp_relocation != 0) {
      half_warp_entry_idx = (__ffs(half_warp_relocation) - 1) % (warpSize/2); // __ffs uses 1-based indexing
    } else if(half_warp_pointer != 0) {
      half_warp_entry_idx = (__ffs(half_warp_pointer) - 1) % (warpSize/2); // __ffs uses 1-based indexing
    } else {
      half_warp_entry_idx = -1; // TODO: is this broken?
    }

    half_warp_entry = __shfl(thread_table_entry, half_warp_entry_idx, warpSize/2);
  }

  __device__ inline void insert_phase_2_determine_stage(i32& half_lane_idx, u32& half_lane_mask, u64& half_warp_temp, i32& stage) {
    if(relocate(half_warp_entry) == 1) {
      // either in stage 2,3, or 4: check linked element
      u64 half_warp_link_entry = ring_buffer[pointer(half_warp_entry)];

      if(relocate(half_warp_link_entry) == 1) {
        half_warp_temp = half_warp_link_entry;
        stage = 4;
      } else {
        // either stage 2 or 3, and we need to search forward to differentiate
        insert_phase_2_determine_stage_search(half_lane_idx, half_lane_mask, half_warp_temp, stage, half_warp_link_entry)
      }
    } else if(half_warp_pointer != 0){
      stage = 1;
    } else {
      stage = 5;
    }

  }

  __device__ inline void insert_phase_2_determine_stage_search(i32& half_lane_idx, u32& half_lane_mask, u64& half_warp_temp, i32& stage, u64& half_warp_link_entry) {
    // Either stage 2 or 3: element has relocation bit, but its first linked element doesn't: find slot relocated element is supposed to go in
    i32 stride = hash_slot(key(half_warp_entry),c,d);
    i32 max_num_lines = GPLDA_HASH_MAX_NUM_LINES - key_distance(key(half_warp_entry), slot);
    for(i32 i = 1; i <= max_num_lines; ++i) {
      // if we're at the last iteration and haven't exited the loop yet, return indicating failure
      if(i == max_num_lines) {
        stage = -1;
        break;
      }

      i32 search_slot = (slot + i * stride) % size;
      u64* address = &data[search_slot + half_lane_idx];
      u64 thread_search_entry = *address;

      // first, check the slot and possible pointers to see if element is there
      u32 found;
      u32 ptr;
      do {
        // if element is found, we are in Stage 3
        found = __ballot(key(half_warp_entry) == key(thread_search_entry)) & half_lane_mask;
        if(found != 0) {
          half_warp_temp = half_warp_link_entry;
          stage = 3;
          break;
        }

        // if pointers are present, follow them and check again
        ptr = false;
        if(pointer(thread_table_insert_entry) != null_pointer()) {
          ptr = true;
          address = &ring_buffer[pointer(thread_search_entry)];
          thread_search_entry = *address;
        }
        ptr = __ballot(ptr) & half_lane_mask;
      } while(ptr != 0);

      // exit if we found an element
      if(found != 0) {
        break;
      }

      // if no pointers, check to see if slot contains an empty element
      u32 slot_empty = __ballot(thread_search_entry == empty()) & half_lane_mask;
      if(slot_empty != 0) {
        half_warp_temp = empty();
        half_warp_temp_idx = search_slot + ((__ffs(slot_empty) - 1) % (warpSize/2));
        stage = 2;
        break;
      }

      // after pointers have been exhausted, check if element should be evicted, and insert into queue
      u32 evict = __ballot(key_distance(key(thread_search_entry), search_slot) < i) & half_lane_mask;
      if(evict != 0) {
        i32 evict_half_lane_idx = (__ffs(evict) - 1) % (warpSize/2);
        half_warp_temp = __shfl(thread_search_entry, evict_half_lane_idx, warpSize/2);
        half_warp_temp_idx = search_slot + evict_half_lane_idx;
        stage = 2;
        break;
      }
    }
  }

  __device__ inline void insert_phase_2_stage_1(i32& slot, u64*& half_warp_address, u64& half_warp_entry, u64& half_warp_new_entry, i32& half_warp_entry_idx) {
    // Stage 1: we have pointers, but no relocation bit: resolve pointer on first thread that found it
    half_warp_address = &data[slot + half_warp_entry_idx];
    half_warp_new_entry = with_relocate(true,half_warp_entry);
  }

  __device__ inline void insert_phase_2_stage_2(u64*& half_warp_address, u64& half_warp_entry, u64& half_warp_new_entry, i32& half_warp_entry_idx, u64& half_warp_temp) {
    if(half_warp_temp == empty()) {
      half_warp_address = &data[half_warp_temp_idx];
      half_warp_new_entry = with_relocate(false,half_warp_entry);
      half_warp_entry = empty();
      half_warp_entry_idx = 0;
    } else {
      // grab slot from ring buffer
      u32 buffer_idx;
      if(half_lane_idx == 0) {
        buffer_idx = ring_buffer_pop();
        ring_buffer[buffer_idx] = with_relocate(false,half_warp_entry);
      }
      buffer_idx = __shfl(buffer_idx, 0, warpSize/2);

      // prepare entry for insertion
      half_warp_address = &data[half_warp_temp_idx];
      half_warp_entry = half_warp_temp; // same as dereferencing
      half_warp_new_entry = with_pointer(buffer_idx, half_warp_temp);
      half_warp_entry_idx = 0;
    }
  }

  __device__ inline void insert_phase_2_stage_2_cleanup(i32& half_lane_idx, u64& half_warp_new_entry) {
    // swap failed: return linked slot to ring buffer
    if(half_lane_idx == 0) {
      ring_buffer_push(pointer(half_warp_new_entry));
    }
  }

  __device__ inline void insert_phase_2_stage_3(i32& slot, u64*& half_warp_address, u64& half_warp_entry, u64& half_warp_new_entry, i32& half_warp_entry_idx, u64& half_warp_temp) {
    // Stage 3: we have a relocation bit, it has been moved forward, but first linked element has no relocation bit: set relocation bit on linked element
    half_warp_address = &ring_buffer[pointer(half_warp_entry)];
    half_warp_entry = half_warp_temp; // same as dereferencing above address
    half_warp_new_entry = with_relocate(true, half_warp_entry);
  }

  __device__ inline void insert_phase_2_stage_4(i32& slot, u64*& half_warp_address, u64& half_warp_entry, u64& half_warp_new_entry, i32& half_warp_entry_idx, u64& half_warp_temp) {
      // Stage 4: first linked element has a relocation bit: remove relocation bit, move it and advance to next slot
      half_warp_address = &data[slot + half_warp_entry_idx];
      half_warp_new_entry = with_relocate(false, half_warp_temp);
  }

  __device__ inline void insert_phase_2_stage_4_advance(i32& half_lane_idx, u32& half_lane_mask, i32& slot, u64& half_warp_entry, u64& half_warp_new_entry) {
    // make sure to return slot to ring buffer
    if(half_lane_idx == 0) {
      ring_buffer_push(pointer(half_warp_entry));
    }

    // advance to next slot, until we find the previously-lined entry's key
    i32 advance_stride = hash_slot(key(half_warp_new_entry), c,d);
    i32 advance_max_num_lines = GPLDA_HASH_MAX_NUM_LINES - key_distance(key(half_warp_new_entry), slot);
    for(i32 i = 1; i < advance_max_num_lines; ++i) {
      i32 advance_slot = (slot + i * advance_stride) % size;
      u64* address = &data[advance_slot + half_lane_idx];
      u64 thread_advance_entry = *address;

      // check slot and possible pointers to see if element is there
      u32 found;
      u32 ptr;
      do {
        // if element is found, set flag, broadcast it, and exit the loop
        found = false;
        if(key(thread_advance_entry) == key(half_warp_new_entry)) {
          found = true;
        }
        found = __ballot(found) & half_lane_mask;

        // if pointers are present, follow them and check again
        ptr = false;
        if(found == 0 && pointer(thread_advance_entry) != null_pointer()) {
          ptr = true;
          address = &ring_buffer[pointer(thread_advance_entry)];
          thread_advance_entry = *address;
        }
        ptr = __ballot(ptr) & half_lane_mask;
        // TODO: ensure last pointer is as expected
      } while(found == 0 && ptr != 0);

      // exit loop if we found the element and set the new slot
      if(found != 0) {
        slot = advance_slot;
        break;
      }

      // TODO: what if something failed and slot is never found?
    }
  }





  __device__ inline void insert_phase_2(i32& half_lane_idx, u32& half_lane_mask, i32& insert_failed, i32& slot, i32& stride) {
    // resolve queue
    u32 finished;
    do {
      finished = false;

      // determine which thread's entry should be handled and broadcast to all threads
      u64* half_warp_address;
      u64 half_warp_entry;
      u64 half_warp_new_entry;
      i32 half_warp_entry_idx;
      u64 half_warp_temp;
      i32 half_warp_temp_idx;
      insert_phase_2_determine_index(half_lane_idx, half_lane_mask, slot, half_warp_address, half_warp_entry, half_warp_entry_idx);

      // determine stage
      i32 stage;
      insert_phase_2_determine_stage(half_lane_idx, half_lane_mask, half_warp_temp, stage);

      // determine CAS target
      if(stage == 1) {
        insert_phase_2_stage_1(slot, half_warp_address, half_warp_entry, half_warp_new_entry, half_warp_entry_idx);
      } else if(stage == 2) {
        insert_phase_2_stage_2(half_warp_address, half_warp_entry, half_warp_new_entry, half_warp_entry_idx, half_warp_temp);
      } else if(stage == 3) {
        insert_phase_2_stage_3(half_warp_address, half_warp_entry, half_warp_new_entry, half_warp_entry_idx, half_warp_temp);
      } else if(stage == 4) {
        insert_phase_2_stage_4(slot, half_warp_address, half_warp_entry, half_warp_new_entry, half_warp_entry_idx, half_warp_temp);
      } else if(stage == 5) {
        // Stage 5: no relocation bit or pointer present, so we must have either inserted to an empty slot or accumulated existing element
        finished = true;
      } else {
        insert_failed = true;
      }

      // perform CAS
      i32 success;
      if(half_lane_idx == half_warp_entry_idx) {
        u64 old = atomicCAS(half_warp_entry_address, half_warp_entry, half_warp_new_entry);
        success = (old == half_warp_entry);
      }
      success = __shfl(success, half_warp_entry_idx, warpSize/2);

      // perform post-CAS operations
      if(stage == 2 && !success) {
        // CAS failed: perform cleanup
        insert_phase_2_stage_2_cleanup(half_lane_idx, half_warp_new_entry);
      } else if(stage == 4 && success) {
        // CAS succeeded: return slot to ring buffer and move to next slot
        insert_phase_2_stage_4_advance(half_lane_idx, half_lane_mask, slot, half_warp_entry, half_warp_new_entry);
      }

      // ensure entire half warp finishes
      finished = __ballot(finished) & half_lane_mask;
    } while(finished == 0);
  }





  __device__ inline i32 attempt_insertion(u32 half_warp_key, i32 diff) {
    // determine half warp indices
    i32 lane_idx = threadIdx.x % warpSize;
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));

    // build entry to be inserted and shuffle to entire half warp
    u64 half_warp_entry = __shfl(entry(false,null_pointer(),half_warp_key,max(0,diff)), 0, warpSize/2);

    // create a variable so that we return only once
    i32 insert_failed = false;

    // insert key into linked queue
    i32 slot = hash_slot(half_warp_key,a,b);
    i32 stride = hash_slot(half_warp_key,c,d);

    if(diff != 0) {
      insert_phase_1(half_warp_key, diff, lane_idx, half_lane_idx, half_lane_mask, half_warp_entry, insert_failed, slot, stride);
    }

    if(diff != 0 && insert_failed == false) {
      insert_phase_2(half_lane_idx, half_lane_mask, insert_failed, slot, stride);
    }

    // return indicating success
    return insert_failed;
  }






  __device__ __forceinline__ void insert2(u32 key, i32 diff) {
    // if resize in progress, join
    join_resize();

    // try to insert
    i32 failure_resize = attempt_insertion(key, diff);

    // if a warp failed due to table being full, trigger resize
    if(__ballot(failure_resize == 1 || failure_resize == 2) != 0) {
      trigger_resize(failure_resize == 1 ? key : empty_key(), failure_resize == 1 ? diff : 0);
    }
  }
};

}
