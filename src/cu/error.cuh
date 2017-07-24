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

#ifdef CUBLAS_API_H_
const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown CUBLAS error";
}

inline void operator>>(cublasStatus_t error, const FileLine &fl) {
  if(error != CUBLAS_STATUS_SUCCESS) /*{*/
    printf("CUDA error: %s %s:%d%s", cublasGetErrorString(error), fl.file, fl.line, "\n");
    /*exit(error);*/
  /*}*/
}
#endif

}

#define GPLDA_CHECK FileLine(__FILE__, __LINE__)

#endif
