#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <mpi.h>
int      _debug;
#include "kmeans.h"
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -r             : output file in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
}

int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j;
           int     isInFileBinary, isOutFileBinary;
           int     is_output_timing, is_print_usage;
           int     numClusters, numCoords, numObjs, totalNumObjs;
           int    *membership;    /* [numObjs] */
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
           int        rank, nproc, mpi_namelen;
           char       mpi_name[MPI_MAX_PROCESSOR_NAME];
           MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Get_processor_name(mpi_name,&mpi_namelen);
    _debug           = 0;
    threshold        = 0.001;
    numClusters      = 0;
    isInFileBinary   = 0;
    isOutFileBinary  = 0;
    is_output_timing = 0;
    is_print_usage   = 0;
    filename         = NULL;
    while ( (opt=getopt(argc,argv,"p:i:n:t:abdorh"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isInFileBinary = 1;
                      break;
            case 'r': isOutFileBinary = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case 'd': _debug = 1;
                      break;
            case 'h': is_print_usage = 1;
                      break;
            default: is_print_usage = 1;
                      break;
        }
    }
    if (filename == 0 || numClusters <= 1 || is_print_usage == 1) {
        if (rank == 0) usage(argv[0], threshold);
        MPI_Finalize();
        exit(1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    objects = mpi_read(isInFileBinary, filename, &numObjs, &numCoords,
                       MPI_COMM_WORLD);
    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;
    MPI_Allreduce(&numObjs, &totalNumObjs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        for (i=0; i<numClusters; i++)
            for (j=0; j<numCoords; j++)
                clusters[i][j] = objects[i][j];
    }
    MPI_Bcast(clusters[0], numClusters*numCoords, MPI_FLOAT, 0, MPI_COMM_WORLD);
    membership = (int*) malloc(numObjs * sizeof(int));
    mpi_kmeans(objects, numCoords, numObjs, numClusters, threshold, membership,
               clusters, MPI_COMM_WORLD);
    free(objects[0]);
    free(objects);
    mpi_write(isOutFileBinary, filename, numClusters, numObjs, numCoords,
              clusters, membership, totalNumObjs, MPI_COMM_WORLD);
    free(membership);
    free(clusters[0]);
    free(clusters);
    MPI_Finalize();
    return(0);
}
