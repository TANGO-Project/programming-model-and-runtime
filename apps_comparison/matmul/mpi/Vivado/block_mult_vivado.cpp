#include <iostream>
#include "matrixmul_zynq.h"
using namespace std;
void multiply_accumulative (double *a, double *b, double *c, int workerRow, int workerColumn, int blockSieze, int matrixSize, int Wj ){
{
   double in_mat_a[blockSize][matrixSize];
   double in_mat_b[matrixSize][Wj];
   double hw_result[blockSize][Wji];
   int i, j, err_cnt = 0;
	for(i = 0; i < blockSize; i++) {
		for(j = 0; j < matrixSize; j++) {
			n_mat_a[i][j] = a[(workerRow+i)*matrixSize+j];
		}
	}
	for(i = 0; i < matrixSize; i++) {
		for(j = 0; j < Wj; j++) {
			in_mat_b[i][j] = b[i*matrixSize+workerColumn +j];
		}
	}
   matrixmul(in_mat_a, in_mat_b, hw_result, blockSize, matrixSize, Wj);
	for(i = 0; i < blockSize; i++) {
		for(j = 0; j < Wj; j++) {
			c[(workerRow+i) * matrixSize + workerColumn +j] = hw_resutl[i][j];
		}
	}
}
