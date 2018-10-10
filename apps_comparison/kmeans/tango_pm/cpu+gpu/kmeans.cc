#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include "kmeans.h"
#include "kmeans_io.h"
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename \n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1) (default 2)\n"
	"	-f num_frags   : number of fragments (must > 1) (default 2)\n"
        "       -l iterations   : number of fragments (must > 1) (default 10)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}
void kmeans( int numClusters, int numFrags, int objsFrag, int loop_iteration, char* filePath, int isBinaryFile, int is_output_timing)
{
    int     numCoords, i, j, index, loop=0;
    double  timing, io_timing, clustering_timing;
    numCoords = file_read_coords(isBinaryFile, filePath);
    numO
    if (numCoords<1){
        fprintf(stderr,"Error reading number of coordinates");
        exit(-1);
    }
    float *clusters = (float*) malloc(numClusters*numCoords*sizeof(float));
    for (i=0; i<numClusters; i++){
        for (j=0; j<numCoords; j++){
            clusters[i*numCoords+j] = (float) ((rand()) / (float)((RAND_MAX/10))-5);
        }
    }
    int **newClusterSize = (int **) malloc(numFrags*sizeof(int*));
    float **newClusters = (float **) malloc(numFrags*sizeof(float*));
    float **fragments = (float **)malloc(numFrags*sizeof(float*));
    for (j=0; j<numFrags; j++){
	newClusterSize[j]=(int*) malloc(numClusters* sizeof(int));
        for (i=0; i<numClusters; i++){
		 newClusterSize[j][i]=0;
    }
    compss_on();
    for (i=0; i<numFrags; i++){
        fragments[i] = init_Fragment(objsFrag, numCoords, filePath);
    }
    do {
        for (i=0; i<numFrags; i++){
	    compute_newCluster(objsFrag, numCoords, numClusters, fragments[i], clusters, newCluster[i], newClusterSize[i]);
            if (i>0){
                merge_newCluster(numCoords, numClusters, newCluster[0], newCluster[i], newClusterSize[0], newClusterSize[i]);
	    }
        }
        compss_wait_on(newClusterSize[0]);
        compss_wait_on(newCluster[0]);
        for (i=0; i<numClusters; i++){
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[0].size[i] > 0){
                    clusters[i*numCoordsj] = newCluster[0][i*numCoords+j] / newClusterSize[0][i];
		}
            }
        }
    } while (loop++ < loop_iteration);
    compss_off();
}

int main(int argc, char **argv) {
    extern char   *optarg;
    extern int     optind;
           int     opt, i, j, isBinaryFile, is_output_timing, numClusters, numFrags, numObjs, loop_iteration;
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
    threshold        = 0.001;
    numClusters      = 2;
    numFrags         = 2;
    numObjs	     = 100000;
    loop_iteration   = 10;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;
    while ( (opt=getopt(argc,argv,"p:i:n:f:l:t:abo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'f': numFrags = atoi(optarg);
                      break;
            case 'l': loop_iteration = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }
    if (filename == 0 || numClusters <= 1 || numFrags < 1 || loop_iteration < 1) usage(argv[0], threshold);
    kmeans(numClusters, numFrags, numObjs, loop_iteration, filename, isBinaryFile, is_output_timing);
    return(0);
}
