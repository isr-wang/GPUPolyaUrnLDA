#pragma once

#include "types.cuh"

namespace gplda {

template<class T>
class DSMatrix {
  public:
    T* dense;
    DSMatrix();
    ~DSMatrix();
};

extern template class DSMatrix<f32>;
extern template class DSMatrix<u32>;

}
