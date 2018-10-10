#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
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

float *init_Fragment(int numObjs, int numCoords, char* filePath, int isBinaryFile){
   float *frag = file_read(false, filePath, numObjs, numCoords);
   return frag;	
}

void compute_newCluster(int numObjs, int numCoords, int numClusters, float *frag, float *clusters, float *newClusters, int *newClustersSize){
	int *index = new int[frag->numObjs]; 
	int i,j;
        for (i=0; i<numObjs; i++) {
           #pragma omp task 
           index[i] = find_nearest_cluster(clusters->numClusters, frag->numCoords, frag->objects[i], clusters->coords);
	}
        #pragma omp taskwait
	for (i=0; i<numObjs; i++) {
            newClustersSize[index[i]]++;
	    for (j=0; j<numCoords; j++){
                if (newClustersSize[index[i]]==1){
			newClusters[index[i]*numCoords+j]= 0.0;
		}
		newClusters[index[i]*numCoords+j] += frag[i*numCoords + j];
	    }
        }
}
void compute_newCluster_GPU(int numObjs, int numCoords, int numClusters, float *frag, float *clusters, float *newClusters, int *newClustersSize){
	cuda_find_nearest_cluster(numCoords, numObjs, numClusters, frag, clusters, newClustersSize);
        #pragma omp taskwait
        for (i=0; i<numObjs; i++) {
            newClustersSize[index[i]]++;
            for (j=0; j<numCoords; j++){
                if (newClustersSize[index[i]]==1){
                        newClusters[index[i]*numCoords+j]= 0.0;
                }
                newClusters[index[i]*numCoords+j] += frag[i*numCoords + j];
            }
        }
}
void merge_newCluster(int numCoords, int numClusters, float *newClusters_1, float *newClusters_2, int *newClustersSize_1, int *newClustersSize_2){
        int i, j;
        for (i=0; i<numClusters; i++){
            newClustersSize_1->size[i] = newClustersSize_1->size[i] + newClustersSize_2->size[i];
            #pragma omp task
            for (j=0; j<numCoords; j++) {
                newClusters_1[i*numCoords+j] += newClusters_2[i*numCoords+j];
            }
        }
        #pragma omp taskwait
}

