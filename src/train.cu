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
  return 0;
}
