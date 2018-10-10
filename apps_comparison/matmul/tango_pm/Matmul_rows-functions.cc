#include "Matmul.h"
void multiplyBlock(double *blockA, double *blockB, double *blockC, int N, int M) {
        for (int i=0; i<N; i++) {
                for (int j=0; j<M; j++) {
                        for (int k=0; k<M; k++) {
                                blockC[i*M+j] += blockA[i*M+k] * blockB[k*M+j];
                        }
                }
        }
}
