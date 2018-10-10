#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <omp.h>
int      _debug;
#include "kmeans.h"
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -p nproc       : number of threads (default system allocated)\n"
        "       -a             : perform atomic OpenMP pragma (default no)\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j, nthreads;
           int     isBinaryFile, is_perform_atomic, is_output_timing;
           int     numClusters, numCoords, numObjs;
           int    *membership;    /* [numObjs] */
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;

    _debug            = 0;
    nthreads          = 0;
    numClusters       = 0;
    threshold         = 0.001;
    numClusters       = 0;
    isBinaryFile      = 0;
    is_output_timing  = 0;
    is_perform_atomic = 0;
    filename          = NULL;

    while ( (opt=getopt(argc,argv,"p:i:n:t:abdo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'p': nthreads = atoi(optarg);
                      break;
            case 'a': is_perform_atomic = 1;
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case 'd': _debug = 1;
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }
    if (filename == 0 || numClusters <= 1) usage(argv[0], threshold);
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);
    membership = (int*) malloc(numObjs * sizeof(int));
    clusters = omp_kmeans(is_perform_atomic, objects, numCoords, numObjs,
                          numClusters, threshold, membership);
    free(objects[0]);
    free(objects);
    file_write(filename, numClusters, numObjs, numCoords, clusters, membership);
    free(membership);
    free(clusters[0]);
    free(clusters);
    return(0);
}

