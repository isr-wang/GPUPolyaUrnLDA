namespace gplda {

__global__ void warp_sampler(size_t size, uint32_t n_docs, uint32_t *z, uint32_t *w, uint32_t *d_len, uint32_t *d_idx);

}