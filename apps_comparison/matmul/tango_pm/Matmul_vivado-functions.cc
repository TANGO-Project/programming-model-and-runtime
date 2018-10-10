#include "Matmul.h"
void multiplyBlock(double *blockA, double *blockB, double *blockC, int M) {
       Muld(blockA,blockB,M,M,blockC); 
	#pragma omp taskwait
}
