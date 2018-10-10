
void multiply_block_accumulative (double *a, double *b, double *c, int workerRow, int workerColumn, int blockSieze, int matrixSize, int Wj){
 #pragma acc data region copyout(c[workerRow*matrixsize+workerColumn:(blockSize*blockSize)-1]), copyin(a[workerRow*matrixSize:(blockSize*matrixSize)-1],c[workerColum:(blockSize*matrixSize)+blockSize-1])
  {
   #pragma acc region for parallel, vector(8)
   for (i = 0; i < blockSize; ++i) {
        #pragma acc region for parallel, vector(8)
        for (k = 0; k < matrixSize; ++k) {
	    double sum = 0.0
	    #pragma acc for seq        
            for (j = 0; j < Wj; ++j) {
               c[(workerRow + i)*matrixSize + workerColumn + j] +=
                    a[(workerRow + i)*matrixSize + k]*b[k*matrixSize + workerColumn + j];
            }
            c[(workerRow + i)*matrixSize + workerColumn + j] = sum;
        }
    }
}
