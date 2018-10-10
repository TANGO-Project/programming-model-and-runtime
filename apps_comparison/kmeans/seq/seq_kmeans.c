#include <stdio.h>
#include <stdlib.h>
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

float** seq_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **newClusters;    /* [numClusters][numCoords] */
    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];
    for (i=0; i<numObjs; i++) membership[i] = -1;
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;
    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);
            if (membership[i] != index) delta += 1.0;
            membership[i] = index;
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);
    *loop_iterations = loop + 1;
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    return clusters;
}

