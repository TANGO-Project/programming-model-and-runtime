#include "Matmul.h"
int N;  //ROWBlocks
int M;	//ROWSIZE
double **A, **C;
double B[N*M*M]
double val;
double *init_row_block(double value){
     double *block = (double*) malloc(M*M); 
     for (int i = 0; i < M*M*N; i++){
	  block[i]=val;
     }
     return block
}
void init_matrices(){
     A = (double**) malloc(N*sizeof(double*);
     C = (double **) malloc(N*sizeof(double*);		
     for (int i = 0; i < N; i++){
               	A[i*N+j]=init_row_block(val);
                C[i*N+j]=init_row_block(0.0);
                for (int k=0; k< M*M; k++){
	             B[i*N+j+k]=val;
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
		multiply_row(A[i], B, C[i], N*M, N*N*M);
       	}
	compss_off();
	return 0;
}
