#include "dsmatrix.cuh"
#include "error.cuh"
#include "train.cuh"

namespace gplda {

template<class T>
DSMatrix<T>::DSMatrix<T>() {
  cudaMalloc/*Pitch*/(&dense, ARGS->K * ARGS->V * sizeof(T)) >> GPLDA_CHECK;
}

template<class T>
DSMatrix<T>::~DSMatrix<T>() {
  cudaFree(dense) >> GPLDA_CHECK;
}

template class DSMatrix<float>;
template class DSMatrix<uint32_t>;

}
