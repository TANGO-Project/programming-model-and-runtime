#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */
void fillMatrix(const char* fileName, int matrixSize, double* mat) {
    int i, j;
    FILE *file;
    file = fopen(fileName, "r");
    for(i = 0; i < matrixSize; i++) {
        for(j = 0; j < matrixSize; j++) {
            if (!fscanf(file, "%lf", &mat[i*matrixSize + j])) {
                break;
            }
        }
    }
    fclose(file);    
}
void multiply_accumulative (double *a, double *b, double *c, int workerRow, int workerColumn, int wi, int Wj, intWk);

int main (int argc, char *argv[]) {   
    int matrixSize = atoi(argv[1]);     // Number of rows/columns in matrix A B and C
    char* ain = argv[2];                // FileName of Ain
    char* bin = argv[3];                // FileName of Bin
    char* cout = argv[4];               // FileName of Cout
    double a[matrixSize*matrixSize];    // Matrix A to be multiplied
    double b[matrixSize*matrixSize];    // Matrix B to be multiplied
    double c[matrixSize*matrixSize];    // Result matrix C    
    // MPI configuration
    int numProcs;                       // Number of MPI Nodes
    int numProcsPerDimension;           // Number of blocks per row/column
    int blockSize;                      // Block size
    int taskId;                         // Task identifier
    MPI_Status status;                  // Status variable for MPI communications
    MPI_Request send_request;           // For async calls
    int mtype;                          // Message type
    int i, j, k, dest, row, workerRow, workerColumn;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    numProcsPerDimension = sqrt(numProcs);
    blockSize = matrixSize / numProcsPerDimension;    
    if (taskId == MASTER) {
        // Initialize arrays
        fillMatrix(ain, matrixSize, a);
        fillMatrix(bin, matrixSize, b);
        fillMatrix(cout, matrixSize, c);
        // Send matrix data to the worker tasks
        mtype = FROM_MASTER;
        for (dest = 1; dest < numProcs; dest++) {
            workerRow = (dest/numProcsPerDimension)*blockSize;
            workerColumn = (dest%numProcsPerDimension)*blockSize;
            // Send block parameters
            MPI_Isend(&workerRow, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD, &send_request);
            MPI_Isend(&workerColumn, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD, &send_request);
            // Send block rows of A
            MPI_Isend(&a[workerRow*matrixSize], matrixSize*blockSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &send_request);
            // Send block columns of B
            for (row = 0; row < matrixSize; ++row) {
                MPI_Isend(&b[row*matrixSize + workerColumn], blockSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &send_request);
            }
            // Send block of C
            for (row = workerRow; row < workerRow + blockSize; ++row) {
                MPI_Isend(&c[row*matrixSize + workerColumn], blockSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &send_request);
            }
        }
    }
    if (taskId > MASTER) {
        // Receive matrix
        mtype = FROM_MASTER;
        MPI_Recv(&workerRow, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&workerColumn, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[workerRow*matrixSize], matrixSize*blockSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        for (row = 0; row < matrixSize; ++row) {
            MPI_Recv(&b[row*matrixSize + workerColumn], blockSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        }
        for (row = workerRow; row < workerRow + blockSize; ++row) {
            MPI_Recv(&c[row*matrixSize + workerColumn], blockSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        }
    } else {
        workerRow = 0;
        workerColumn = 0;
    }
    // Perform multiply accumulative
    multiply_accumulative (a, b, c, workerRow,  workerColumn, blockSieze, matrixSize, blockSize);
    // Send back result to master
    mtype = FROM_WORKER;
    for (row = workerRow; row < workerRow + blockSize; ++row) {
        MPI_Isend(&c[row*matrixSize + workerColumn], blockSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &send_request);
    }
    if (taskId == MASTER) {
        // Receive results from worker tasks
        mtype = FROM_WORKER;
        for (dest = 0; dest < numProcs; dest++) {
            workerRow = (dest/numProcsPerDimension)*blockSize;
            workerColumn = (dest%numProcsPerDimension)*blockSize;
            // Receive C block
            for (row = workerRow; row < workerRow + blockSize; ++row) {
                MPI_Recv(&c[row*matrixSize + workerColumn], blockSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &status);
            }
        }
    }
    MPI_Finalize();
}
