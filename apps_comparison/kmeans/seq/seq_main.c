#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
int      _debug;
#include "kmeans.h"
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j;
           int     isBinaryFile, is_output_timing;
           int     numClusters, numCoords, numObjs;
           char   *filename;
           float **clusters;      /* [numClusters][numCoords] cluster center */
           int     loop_iterations;
    _debug           = 0;
    threshold        = 0.001;
    numClusters      = 0;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;
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
    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);
    clusters = _kmeans(filesPath, isBinaryFile, numClusters);
    free(objects[0]);
    free(objects);
    file_write(filename, numClusters, numObjs, numCoords, clusters,
               membership);
    return(0);
}
