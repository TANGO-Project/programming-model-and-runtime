#include <stdio.h>
#define BLOCK_SIZE 32
#pragma omp target device(cuda) copy_deps ndrange(2, 64, 64, 32, 32)
#pragma omp task in(A[0:wA*WA], B[0:wB*wB]) inout(C[0:WB*WA])
__global__ void Muld(double* A, double* B, int wA, int wB, double* C);

