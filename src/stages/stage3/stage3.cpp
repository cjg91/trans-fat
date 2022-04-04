#include <cstdint>
#include <algorithm>
#include "pipeline.hpp"
#include "stage3.hpp"

/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/
void linear_sw(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // Initialize accumulator
            out[i*M+j] = bias[j];
            for (int k = 0; k < K; k++) {
                out[i*M+j] += A[i*K+k] * B[k*M+j];
            }
        }
    }
}

void gelu_sw(int32_t* gelu_in, int32_t* gelu_out, int rows, int cols, float scaling_factor) {

    // TODO: Implement scaling_factor / k for int_erf scaling factor!
    float k = 1.4142;
    int constant = 14;
    float coef_0 = -0.2888;
    float coef_1 = -1.769;
    float coef_2 = 1/coef_0;

    // int_erf
    int b_int = int(coef_1 / scaling_factor);
    int c_int = int(coef_2 / (scaling_factor*scaling_factor));

    float sigmoid_scaling_factor = (scaling_factor * scaling_factor * coef_0)*(1 << constant);
    int32_t shift_int = int(1 / sigmoid_scaling_factor);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
        int32_t val = gelu_in[i*cols+j];
        int32_t sign = (val >= 0) ? 1 : -1;   
        int32_t val_abs = val * sign;
        int32_t abs_int = std::min(val_abs, -1*b_int);
        int32_t intermediate = (abs_int + b_int);
        int32_t y_int = sign * (intermediate * intermediate + c_int);
        y_int = y_int / (1 << constant);

        val = val * (y_int + shift_int)


        }
    }
}


void requantize(int32_t* in, int8_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = int8_t(in[i*cols+j] * M_scale);
        }
    }
}


void stage3_gt(int8_t* fc_in, int8_t* dense_weight_t, int32_t* dense_bias, int8_t* dense_out, float dense_acc_scale, float M_stage3) {

    auto dense_temp = new int32_t[SEQLEN*FFDIM];
    auto gelu_temp = new int32_t[SEQLEN*FFDIM];

    linear_sw(fc_in, dense_weight_t, dense_bias, dense_temp, SEQLEN, FFDIM, DMODEL);

    gelu_sw(dense_temp, gelu_temp, SEQLEN, FFDIM, dense_acc_scale);

}