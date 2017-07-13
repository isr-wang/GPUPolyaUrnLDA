#ifndef GPLDA_DSMATRIX_H
#define GPLDA_DSMATRIX_H

#include "stdint.h"

namespace gplda {

template<class T>
class DSMatrix {
  public:
    T* dense;
    DSMatrix();
    ~DSMatrix();
};

extern template class DSMatrix<float>;
extern template class DSMatrix<uint32_t>;

}

#endif
