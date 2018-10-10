#define N 1024
#define M 1024
#define P 1024

double A[N][P]; // op 1
double B[P][M]; // op 2
double C[N][M]; // res

int main() {
  
  unsigned long i, j, k;

  for (i = 0; i < N; i++){
    for (k = 0; k < P; k++){
      A[i][k] = 0.;
    }
  }
  for (k = 0; k < P; k++){
    for (j = 0; j < M; j++){
      B[k][j] = 0.;
    }
  }
  for (i = 0; i < N; i++){
    for (j = 0; j < M; j++){
      C[i][j] = 0.;
    }
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      for (k = 0; k < P; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return 0;
}
