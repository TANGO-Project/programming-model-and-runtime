#include "mpi.h"
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
    double a[matrixSize*matrixSize];   // Matrix A to be multiplied
    double b[matrixSize*matrixSize];   // Matrix B to be multiplied
    double c[matrixSize*matrixSize];   // Result matrix C    
    int mpiProcs;                       // Number of MPI Nodes
    int taskid;                         // Task identifier
    MPI_Status status;                  // Status variable for MPI communications
    MPI_Request send_request;           // For async calls
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiProcs);
    int i, j, k, dest, rows;
    if (taskid == MASTER) {
        // Initialize arrays
        fillMatrix(ain, matrixSize, a);
        fillMatrix(bin, matrixSize, b);
        fillMatrix(cout, matrixSize, c);
        // Send matrix data to the worker tasks
        int averow = matrixSize/mpiProcs;
        int extra = matrixSize%mpiProcs;
        int offset = 0;
        int mtype = FROM_MASTER;
        for (dest = 0; dest < mpiProcs; dest++) {
            rows = (dest < extra) ? averow+1 : averow;   	
            MPI_Isend(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD, &send_request);
            MPI_Isend(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD, &send_request);
            MPI_Isend(&a[offset*matrixSize], rows*matrixSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &send_request);
            MPI_Isend(&b, matrixSize*matrixSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &send_request);
            MPI_Isend(&c, rows*matrixSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, &send_request);
            offset = offset + rows;
        }
    }
    // Receive matrix
    int offset = 0;
    int mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*matrixSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, matrixSize*matrixSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&c[offset*matrixSize], rows*matrixSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    // Perform multiply accumulative
     multiply_accumulative (a, b, c, 0,  0, rows, matrixSize, matrixSize);
    // Send back result to master
    mtype = FROM_WORKER;
    MPI_Isend(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &send_request);
    MPI_Isend(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &send_request);
    MPI_Isend(&c, rows*matrixSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &send_request);
    if (taskid == MASTER) {
        // Receive results from worker tasks
        mtype = FROM_WORKER;
        for (i = 0; i < mpiProcs; i++) {
            MPI_Recv(&offset, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset*matrixSize], rows*matrixSize, MPI_DOUBLE, i, mtype, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Finalize();
}
