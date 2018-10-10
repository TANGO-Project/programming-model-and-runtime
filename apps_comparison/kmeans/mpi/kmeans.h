#ifndef _H_KMEANS
#define _H_KMEANS
#include <mpi.h>
extern int _debug;
int     mpi_kmeans(float**, int, int, int, float, int*, float**, MPI_Comm);
float** mpi_read(int, char*, int*, int*, MPI_Comm);
int     mpi_write(int, char*, int, int, int, float**, int*, int, MPI_Comm);
#endif
