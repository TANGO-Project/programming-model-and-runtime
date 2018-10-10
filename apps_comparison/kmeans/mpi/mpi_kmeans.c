#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "kmeans.h"

__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;
    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
    return(ans);
}
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);
    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}
int mpi_kmeans(float    **objects,     /* in: [numObjs][numCoords] */
               int        numCoords,   /* no. coordinates */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *membership,  /* out: [numObjs] */
               float    **clusters,    /* out: [numClusters][numCoords] */
               MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    float  **newClusters;    /* [numClusters][numCoords] */
    for (i=0; i<numObjs; i++) membership[i] = -1;
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;
    MPI_Allreduce(&numObjs, &total_numObjs, 1, MPI_INT, MPI_SUM, comm);
    do {
        double curT = MPI_Wtime();
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);
            if (membership[i] != index) delta += 1.0;
            membership[i] = index;
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }
        MPI_Allreduce(newClusters[0], clusters[0], numClusters*numCoords,
                      MPI_FLOAT, MPI_SUM, comm);
        MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT,
                      MPI_SUM, comm);
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (clusterSize[i] > 1)
                    clusters[i][j] /= clusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, comm);
        delta = delta_tmp / total_numObjs;
    } while (delta > threshold && loop++ < 500);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    free(clusterSize);
    return 1;
}
