#include <cstdint>

namespace gplda {

struct Args {
  float alpha;
  float beta;
  uint32_t K;
};

struct Buffer {
  size_t size;
  uint32_t *z;
  uint32_t *w;
  uint32_t *d_len;
  uint32_t *d_idx;
  size_t n_docs;
  uint32_t *gpu_z;
  uint32_t *gpu_w;
  uint32_t *gpu_d_len;
  uint32_t *gpu_d_idx;
};

extern "C" void initialize(Args *args, Buffer *buffers, size_t n_buffers);
extern "C" void sample_phi();
extern "C" void sample_z(Buffer *buffers);
extern "C" void cleanup(Buffer *buffers, size_t n_buffers);

}
