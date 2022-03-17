
#include <stdlib.h>
#include <math.h>

#define TEST_LEN 2048
#define X3_CONST 0.044715

typedef float activation_t


activation_t GELU(const activation_t x) 
{
  #pragma HLS INLINE OFF

  activation_t th = sqrtf(2.0/M_PI)*(x + X3_CONST*powf(x,3));
  activation_t gx = 0.5*x*(1 + tanhf(th));

  return gx;
}

extern "C"
{
void driverGELU(const activation_t *X, activation_t *GX)
{
  #pragma HLS interface m_axi port = X offset = slave bundle = gmem
  #pragma HLS interface m_axi port = GX offset = slave bundle = gmem
  #pragma HLS interface s_axilite port = X bundle = control
  #pragma HLS interface s_axilite port = GX bundle = control

  activation_t X_buff[TEST_LEN];
  activation_t GX_buff[TEST_LEN];

  read_X_loop: for (int i = 0; i < TEST_LEN; i++)
  {
    X_buff[i] = X[i];
  }

  compute_GX_loop: for (int i = 0; i < TEST_LEN; i++)
  {
    GX_buff[i] = GELU(X_buff[i]);
  }

  read_X_loop: for (int i = 0; i < TEST_LEN; i++)
  {
    GX[i] = GX_buff[i];
  }

}
}