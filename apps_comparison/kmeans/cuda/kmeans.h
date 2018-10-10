#ifndef _H_KMEANS
#define _H_KMEANS
#include "kmeans_io.h"
#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}
inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif
float** cuda_kmeans(float**, int, int, int, float, int*, int*);
#endif
