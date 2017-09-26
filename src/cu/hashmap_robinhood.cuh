#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#define GPLDA_HASH_LINE_SIZE 16
#define GPLDA_HASH_MAX_NUM_LINES 4
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
  u32 needs_rebuild;
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

  __device__ __forceinline__ u32 value(u64 entry) {
    return bfe_b64(entry, 0, 36);
  }

  __device__ __forceinline__ static constexpr u32 null_pointer() {
    return 0x7f;
  }

  __device__ __forceinline__ static constexpr u32 empty_key() {
    return 0xfffff;
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

      needs_rebuild = 0;
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





  __device__ inline void rebuild() {

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
          return __shfl(value(entry), __ffs(key_found), warpSize/2);
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





  __device__ inline i32 try_accumulate2(u32 half_warp_key, i32 diff) {
    // determine half warp indices
    i32 lane_idx = threadIdx.x % warpSize;
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / (warpSize / 2)) * (warpSize / 2));

    // build entry to be inserted and shuffle to entire half warp
    u64 half_warp_entry = __shfl(entry(false,null_pointer(),half_warp_key,diff), 0, warpSize/2);

    // create a variable so that we return only once
    i32 insert_failed = false;

    // insert key into linked queue
    i32 slot = hash_slot(half_warp_key,a,b);
    i32 stride = hash_slot(half_warp_key,c,d);

    debug_print_slot(slot,0,"before");
    debug_print_slot(slot,16,"before");

    if(insert_failed == false) {
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
              half_warp_write_entry = with_value(value(half_warp_entry) + value(thread_table_entry), half_warp_entry);
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
          insert_failed = true;
        }
      }
    }

    __syncthreads();
    debug_print_slot(slot,0,"after");
    debug_print_slot(slot,16,"after");


    if(insert_failed == false) {
      // resolve queue
      u32 finished;
      do {
        // find element to be resolved
        u64 thread_table_entry = data[slot + half_lane_idx];
        finished = false;

        u32 half_warp_relocation = __ballot(relocate(thread_table_entry) != 0) & half_lane_mask;
        u32 half_warp_pointer = __ballot(pointer(thread_table_entry) != null_pointer()) & half_lane_mask;
        if(half_warp_relocation != 0) {
          // resolve relocation bit: first, broadcast pointer to entire half warp, then retrieve entry
          u32 lane_link_entry_idx = __ffs(half_warp_relocation) - 1;
          u32 half_warp_link_entry_pointer = __shfl(pointer(thread_table_entry), lane_link_entry_idx % (warpSize/2), warpSize/2);
          u64 half_warp_link_entry = ring_buffer[half_warp_link_entry_pointer];

          // figure out whether linked element should take thread's slot, or whether thread's slot needs to be moved
          if(relocate(half_warp_link_entry) == 1) {
            // first linked element has a relocation bit: move it
            if(lane_idx == lane_link_entry_idx) {
              // no need to check for success: whether we succeed or fail, try again and keep going
              atomicCAS(&data[slot + half_lane_idx], thread_table_entry, half_warp_link_entry);
            }
          } else {
            // element has relocation bit, but its first linked element doesn't: find slot relocated element is supposed to go in
            u64 half_warp_table_entry;
            half_warp_table_entry = __shfl(thread_table_entry, lane_link_entry_idx % (warpSize/2), warpSize/2);

            // find slot relocated element is supposed to go into
            i32 insert_stride = hash_slot(key(half_warp_table_entry),c,d);
            i32 insert_max_num_lines = GPLDA_HASH_MAX_NUM_LINES - key_distance(key(half_warp_table_entry), slot);
            for(i32 i = 1; i <= insert_max_num_lines; ++i) {
              // if we're at the last iteration and haven't exited the loop yet, return indicating failure
              if(i == insert_max_num_lines) {
                insert_failed = true;
                break;
              }

              i32 insert_slot = (slot + i * insert_stride) % size;
              u64 thread_table_insert_entry = data[insert_slot + half_lane_idx];

              // check first if slot contains an empty element: if so, insert the element there - no need to check pointers because they must be null
              u32 slot_empty = __ballot(thread_table_insert_entry == empty()) & half_lane_mask;
              if(slot_empty != 0) {
                i32 slot_empty_lane_idx = __ffs(slot_empty) - 1;
                if(lane_idx == slot_empty_lane_idx) {
                  u64 thread_new_entry = with_relocate(0,with_pointer(null_pointer(), thread_table_entry));
                  u64 old_entry_int_repr = atomicCAS(&data[insert_slot + half_lane_idx], thread_table_insert_entry, thread_new_entry);
                  if(old_entry_int_repr == empty()) {
                    slot = insert_slot;
                  }
                }
                // ensure entire half warp knows the new slot value, if it changed
                slot = __shfl(slot, slot_empty_lane_idx % (warpSize/2), warpSize/2);
                break;
              }

              // assuming slot is full, check pointers to see if element is there
              u32 found;
              u32 ptr;
              u64* address = &data[insert_slot + half_lane_idx];
              do {
                // if element is found, set relocation bit on its first link
                found = false;
                if(thread_table_insert_entry == half_warp_table_entry) {
                  found = true;
                  u64 half_warp_link_entry_with_relocate = with_relocate(1, half_warp_link_entry);
                  // no need to check for success: whether we succeed or fail, try again and keep going
                  atomicCAS(&ring_buffer[half_warp_link_entry_pointer], half_warp_link_entry, half_warp_link_entry_with_relocate);
                }
                found = __ballot(found == true) & half_lane_mask;

                // if pointers are present, follow them and check again
                ptr = false;
                if(found == 0 && pointer(thread_table_insert_entry) != null_pointer()) {
                  ptr = true;
                  address = &ring_buffer[pointer(thread_table_insert_entry)];
                  thread_table_insert_entry = *address;
                }
                ptr = __ballot(ptr == true) & half_lane_mask;
              } while(found == 0 && ptr != 0);

              // exit if we found an element
              if(found != 0) {
                break;
              }

              // after pointers have been exhausted, check if element should be evicted, and insert into queue
              u32 evict = __ballot(key_distance(key(thread_table_insert_entry), insert_slot) < i) & half_lane_mask;
              if(evict != 0 && lane_idx == __ffs(evict) - 1) {
                // grab slot from ring buffer
                u32 buffer_idx = ring_buffer_pop();
                ring_buffer[buffer_idx] = half_warp_table_entry;

                // prepare entry for insertion
                u64 thread_table_insert_entry_with_pointer = with_pointer(buffer_idx, thread_table_insert_entry);

                // insert entry, returning value to ring buffer if insert failed
                u64 old_entry = atomicCAS(address, thread_table_insert_entry, thread_table_insert_entry_with_pointer);
                if(old_entry != thread_table_insert_entry) {
                  ring_buffer_push(buffer_idx);
                }
              }

              // exit if we evicted
              if(evict != 0) {
                break;
              }
            }
          }
        } else if(half_warp_pointer != 0){
          // we have pointers, but no relocation bit: resolve pointer on first thread that found it
          if(lane_idx == __ffs(half_warp_pointer) - 1) {
            // set relocation bit
            u64 thread_new_entry = with_relocate(1,thread_table_entry);

            // no need to check for success: whether we succeed or fail, try again and keep going
            atomicCAS(&data[slot + half_lane_idx], thread_table_entry, thread_new_entry);
          }
        } else {
          // no relocation bit or pointer present, so we must have either inserted to an empty slot or accumulated existing element
          finished = true;
        }

        // ensure entire half warp finishes
        finished = __ballot(finished) & half_lane_mask;
      } while(finished == 0);
    }

    // return indicating success
    return insert_failed ? 0 : 1;
  }

  __device__ __forceinline__ void accumulate2(u32 key, i32 diff) {
    // try to accumulate
    volatile i32 success = try_accumulate2(key, diff);

    // rebuild if too large
//    sync();
//    if(needs_rebuild == 1) {
//      rebuild();
//    }
  }
};

}
