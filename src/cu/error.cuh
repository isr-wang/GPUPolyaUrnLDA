#ifndef GPLDA_ERROR_H
#define GPLDA_ERROR_H

#include <cstdio>

namespace gplda {

struct FileLine {
    const char* file;
    int line;
    FileLine(const char* f, int l) : file(f), line(l) {}
};

inline void operator>>(cudaError_t error, const FileLine &fl) {
  if(error != cudaSuccess) /*{*/
    printf("CUDA error: %s %s:%d%s", cudaGetErrorString(error), fl.file, fl.line, "\n");
    /*exit(error);*/
  /*}*/
}

}

#define GPLDA_CHECK FileLine(__FILE__, __LINE__)

#endif
