void multiply_block_accumulative (double *a, double *b, double *c, int workerRow, int workerColumn, int blockSieze, int matrixSize, int Wj ){
  for (i = 0; i < blockSize; ++i) {
        for (k = 0; k < matrixSize; ++k) {
            for (j = 0; j < Wj; ++j) {
                c[(workerRow + i)*matrixSize + workerColumn + j] = c[(workerRow + i)*matrixSize + workerColumn + j] +
                    a[(workerRow + i)*matrixSize + k]*b[k*matrixSize + workerColumn + j];
            }
        }
    }
}
