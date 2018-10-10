#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
#include "kmeans.h"
#define MAX_CHAR_PER_LINE 128
float** file_read(int   isBinaryFile,  /* flag: 0 or 1 */
                  char *filename,      /* input file name */
                  int  *numObjs,       /* no. data objects (local) */
                  int  *numCoords)     /* no. coordinates */
{
    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;
    if (isBinaryFile) {  /* input file is in raw binary format -------------*/
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
        numBytesRead = read(infile, numObjs,    sizeof(int));
        assert(numBytesRead == sizeof(int));
        numBytesRead = read(infile, numCoords, sizeof(int));
        assert(numBytesRead == sizeof(int));
        if (_debug) {
            printf("File %s numObjs   = %d\n",filename,*numObjs);
            printf("File %s numCoords = %d\n",filename,*numCoords);
        }
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);
        numBytesRead = read(infile, objects[0], len*sizeof(float));
        assert(numBytesRead == len*sizeof(float));
        close(infile);
    }
    else {  /* input file is in ASCII format -------------------------------*/
        FILE *infile;
        char *line, *ret;
        int   lineLen;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);
        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            while (strlen(line) == lineLen-1) {
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);
                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);
                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }
            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);
        if (_debug) printf("lineLen = %d\n",lineLen);
        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break; /* this makes read from 1st object */
            }
        }
        rewind(infile);
        if (_debug) {
            printf("File %s numObjs   = %d\n",filename,*numObjs);
            printf("File %s numCoords = %d\n",filename,*numCoords);
        }
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);
        i = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++)
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }
        fclose(infile);
        free(line);
    }
    return objects;
}
int file_write(char      *filename,     /* input file name */
               int        numClusters,  /* no. clusters */
               int        numObjs,      /* no. data objects */
               int        numCoords,    /* no. coordinates (local) */
               float    **clusters,     /* [numClusters][numCoords] centers */
               int       *membership)   /* [numObjs] */
{
    FILE *fptr;
    int   i, j;
    char  outFileName[1024];
    sprintf(outFileName, "%s.cluster_centres", filename);
    printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
           numClusters, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numCoords; j++)
            fprintf(fptr, "%f ", clusters[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);
    sprintf(outFileName, "%s.membership", filename);
    printf("Writing membership of N=%d data objects to file \"%s\"\n",
           numObjs, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numObjs; i++)
        fprintf(fptr, "%d %d\n", i, membership[i]);
    fclose(fptr);
    return 1;
}
