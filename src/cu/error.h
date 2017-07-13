#ifndef GPLDA_ERROR_H
#define GPLDA_ERROR_H

#include <cstdio>

struct FileLine {
    const char* file;
    int line;
};

inline FileLine check_error(const char* file, int line) {
  FileLine fl = {file, line};
  return fl;
}

inline void operator >> (cudaError_t error, const FileLine &fl) {
  if(error != cudaSuccess) {
    printf("CUDA error: %s %s:%d%s", cudaGetErrorString(error), fl.file, fl.line, "\n");
    exit(error);
  }
}

#define GPLDA_CHECK check_error(__FILE__, __LINE__)

#endif
