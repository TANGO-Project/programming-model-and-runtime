
void multiply_block_accumulative (double *a, double *b, double *c, int workerRow, int workerColumn, int blockSieze, int matrixSize){
    const int ab_size = blockSize * matrixSize * sizeof(float);
    const int c_size = blockSize * blockSize * sizeof(float);
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, ab_size);
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, ab_size);
    float* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, c_size);
    cuerr = cudaMemcpy(adev, a, ab_size, cudaMemcpyHostToDevice);
    cuerr = cudaMemcpy(bdev, b, ab_size, cudaMemcpyHostToDevice);
    const dim3 threads(blockSize, matrix_size);
    const dim3 blocks( n/threads.x, n/threads.y);
    kernel<<<blocks, threads>>>(adev, bdev, n, cdev);
    cuerr = cudaDeviceSynchronize();
    cuerr = cudaMemcpy(c, cdev, c_size, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    return 0;
}
} 
