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
inline const char* cublasGetErrorString(cublasStatus_t status) {
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
  return "unknown cuBLAS error";
}

inline void operator>>(cublasStatus_t error, const FileLine &fl) {
  if(error != CUBLAS_STATUS_SUCCESS) /*{*/
    printf("cuBLAS error: %s %s:%d%s", cublasGetErrorString(error), fl.file, fl.line, "\n");
  /*exit(error);*/
  /*}*/
}
#endif

#ifdef CURAND_H_
inline const char* curandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "unknown cuRAND error";
}

inline void operator>>(curandStatus_t error, const FileLine &fl) {
  if(error != CURAND_STATUS_SUCCESS) /*{*/
    printf("cuRAND error: %s %s:%d%s", curandGetErrorString(error), fl.file, fl.line, "\n");
  /*exit(error);*/
  /*}*/
}
#endif

}

#define GPLDA_CHECK FileLine(__FILE__, __LINE__)

#endif
