#include "train.h"
#include "alias.h"
#include "dlhmatrix.h"
#include "warpsampler.h"

namespace gplda {

Args ARGS;
DLHMatrix Phi;
DLHMatrix n;
Poisson pois;

void initialize(Args *args, Buffer *buffers, size_t n_buffers) {
  ARGS = *args;
}

void sample_phi() {
}

void sample_z(Buffer *buffer) {
  // copy memory
  warp_sampler(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
}

void cleanup() {
}

} // namespace gplda ends here

int main(void) {
  using namespace gplda;
  Args args = {0.1,0.1,10};
  uint32_t z[5] = {0,0,0,0,0};
  uint32_t w[5] = {1,2,3,4,5};
  uint32_t d_len[5] = {3,2,0,0,0};
  uint32_t d_idx[5] = {0,3,0,0,0};
  Buffer buffer = {5, z, w, d_len, d_idx, 2, NULL, NULL, NULL, NULL};
  initialize(&args, &buffer, 1);
  sample_phi();
  sample_z(&buffer);
  cleanup();
  return 0;
}
