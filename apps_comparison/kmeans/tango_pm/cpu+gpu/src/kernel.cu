#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
static inline int nextPowerOfTwo(int n) {
    n--;
    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints
    return ++n;
}
 __host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{   
    int i;
    float ans=0.0;
    for (i = 0; i < numCoords; i++) { 
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }
    return(ans);
}
__global__ 
void cuda_find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
			  float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership
			)
{
    float *clusters = deviceClusters;
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);
	__syncthreads();
        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            if (objectId == 0) {
	    }
            if (dist < min_dist) { // find the min and its array index 
                min_dist = dist;
                index    = i;
            }
        }
	__syncthreads();
        membership[objectId] = index;
    }
}
