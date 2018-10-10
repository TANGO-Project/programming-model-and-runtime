#include "matrixmul_zynq.h"
using namespace hls;
void matrixmul(double *a, double *b, *c, int sa, int sb , int sj){
int const FACTOR_A = sa/2;
int const FACTOR_B = sb/2;
#pragma HLS INLINE off
#pragma HLS array_partition variable=a block factor=FACTOR_A dim=2
#pragma HLS array_partition variable=b block factor=FACTOR_B dim=1
  double accum;
  Row: for(int i = 0; i < sa; i++) {
    Col: for(int j = 0; j < sj; j++) {
#pragma HLS PIPELINE II=1
      accum = 0;
      Prod: for(int k = 0; k < sb; k++) {
        accum += a[i*sa + k] * b[k*sb+j];
        res[i*sa + j] = accum; //if (k == (MAT_B_ROWS-1)) res[i][j] = accum;
      }
    }
  }
}
