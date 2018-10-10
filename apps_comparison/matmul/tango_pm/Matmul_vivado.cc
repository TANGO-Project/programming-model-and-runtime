#include "Matmul.h"
int N;  //MSIZE
int M;	//BSIZE
double **A, **B, **C;
double val;
double *init_block(double value){
     double *block = (double*) malloc(M*M); 
     for (int i = 0; i < M*M; i++){
	  block[i]=val;
     }
}
void init_matrices(){
	A=(double**)malloc(N*N*sizeof(double*));
        B=(double**)malloc(N*N*sizeof(double*));
        C=(double**)malloc(N*N*sizeof(double*));
        for (int i = 0; i < N; i++){
        	for (int j = 0; j < N; j++){
                	A[i*N+j]=init_block(val);
                        B[i*N+j]=init_block(val);
                        C[i*N+j]=init_block(0.0); 
                }
        }
}
int main(int argc, char **argv) {
	N = atoi(argv[1]);
	M = atoi(argv[2]);
	val = atof(argv[3]);
	compss_on();
	init_matrices();
	for (int i=0; i<N; i++) {
               	for (int j=0; j<N; j++) {
                       	for (int k=0; k<N; k++) {
				multiplyBlocks(A[i*N+k], B[k*N+j], C[i*N+j], M);
                       	}
               }
        }
	compss_off();
	return 0;
}
