#include "train.h"
#include "alias.h"
#include "dlhmatrix.h"
#include "polyaurnsampler.h"
#include "warpsampler.h"

namespace gplda {

Args ARGS;
DLHMatrix *Phi;
DLHMatrix *n;
Poisson *pois;

void initialize(Args *args, Buffer *buffers, size_t n_buffers) {
  ARGS = *args; // copy from Rust
  for(int i = 0; i < n_buffers; ++i) {
    // allocate memory for GPU buffer
  }
  Phi = new DLHMatrix();
  n = new DLHMatrix();
  pois = new Poisson();
}

void cleanup(Buffer *buffers, size_t n_buffers) {
  // no need to delete ARGS: it is static
  for(int i = 0; i < n_buffers; ++i) {
    // deallocate memory for GPU buffer
  }
  delete Phi;
  delete n;
  delete pois;
}

void sample_phi() {
  polya_urn_sampler/*<<<1,1>>>*/();
}

void sample_z(Buffer *buffer) {
  // copy memory
  warp_sampler/*<<<1,1>>>*/(buffer->size, buffer->n_docs, buffer->gpu_z, buffer->gpu_w, buffer->gpu_d_len, buffer->gpu_d_idx);
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
  cleanup(&buffer, 1);
  return 0;
}
