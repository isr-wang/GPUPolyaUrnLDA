#include "warpsample.h"

namespace gplda {

__global__ void warp_sample(size_t size, uint32_t n_docs, uint32_t *z, uint32_t *w, uint32_t *d_len, uint32_t *d_idx) {
  // load current row of Phi into shared memory
  // load Alias table into shared memory (worth it? we may not access it at all)
  // compute
}

}
