#include <sys/types.h>
#include <inttypes.h>
#include <algorithm>
#include <stdio.h>
#include "config.hpp"
#include "stage3.hpp"
#include <iostream>
/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/
void linear_sw3(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {
    
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

    /*
    C++ impl of tensor_quant_gelu. 
    Key differences:
    1. This function takes in int32 and a scaling factor, whereas tensor_quant_gelu takes in float and determines the int32 and scaling_factor.
    2. tensor_quant_gelu returns int32. This op can be fused with requantization to return int8. 
    
    */

    float k = 1.4142;
    int constant = 14;
    float coef_0 = -0.2888;
    float coef_1 = -1.769;
    float coef_2 = 1/coef_0;

    // int_erf
    float int_erf_scaling = scaling_factor / k;
    int32_t b_int = int32_t(coef_1 / int_erf_scaling);
    int32_t c_int = int32_t(coef_2 / (int_erf_scaling*int_erf_scaling));
    float sigmoid_scaling_factor = int_erf_scaling * int_erf_scaling * coef_0;
    sigmoid_scaling_factor = sigmoid_scaling_factor * (1<<constant);

    // TODO, why isn't this value used? Remove if not needed
    //float out_scaling_factor = scaling_factor * sigmoid_scaling_factor / 2;

    int32_t shift_int = int32_t(1 / sigmoid_scaling_factor);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
        int32_t val = gelu_in[i*cols+j];
        int32_t sign = (val >= 0) ? 1 : -1;   
        int32_t val_abs = val * sign;
        int32_t abs_int = std::min(val_abs, -1*b_int);
        int32_t intermediate = (abs_int + b_int);
        int32_t y_int = sign * (intermediate * intermediate + c_int);
        int32_t sigmoid_int = y_int / (1 << constant);

        val = val * (sigmoid_int + shift_int);

        gelu_out[i*cols+j] = val;
        }
    }
}


void requantize2(int32_t* in, int8_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = int8_t(in[i*cols+j] * M_scale);
        }
    }
}

extern "C" {
void stage3_gt(int8_t* fc_in, int8_t* dense_weight_t, int32_t* dense_bias, int8_t* dense_out, float dense_acc_scale, float M_stage3) {
    /*
    High level: inputs are int8_t, output is int8_t. The linear layer goes from int8 -> int32. then, apply GeLU (which takes scaling_factor 
    that is related to the int32_t quantization) which goes from int32 -> int32, then requantize2 to int8. This can all be fused.

    dense_acc_scale:    the scaling factor used within I-GeLU
    M_stage 3:          the requantization factor used to quantize the output of GeLU from 32 to 8 bits.
    */

    // Refactor to fuse layers and not dynamically allocate these
    auto dense_temp = new int32_t[CFG::seqlen*CFG::ffdim];
    auto gelu_temp = new int32_t[CFG::seqlen*CFG::ffdim];


    linear_sw3(fc_in, dense_weight_t, dense_bias, dense_temp, CFG::seqlen, CFG::ffdim, CFG::dmodel);
    gelu_sw(dense_temp, gelu_temp, CFG::seqlen, CFG::ffdim, dense_acc_scale);
    requantize2(gelu_temp, dense_out, CFG::seqlen, CFG::ffdim, M_stage3);
    
    delete [] dense_temp;
    delete [] gelu_temp;
}
}
/**** ^^ Ground Truth Above ^^ ****/

int8_t gelu_fused(int32_t gelu_in, float scaling_factor, float M_stage3, int b_int, int c_int, int shift_int)
{

    /*
    C++ impl of tensor_quant_gelu.
    Key differences:
    1. This function takes in int32 and a scaling factor, whereas tensor_quant_gelu takes in float and determines the int32 and scaling_factor.
    2. tensor_quant_gelu returns int32. This op can be fused with requantization to return int8.

    */
    const int constant = 14;

    int32_t sign = (gelu_in >= 0) ? 1 : -1;
    int32_t val_abs = gelu_in * sign;
    int32_t abs_int = std::min(val_abs, -1 * b_int);
    int32_t intermediate = (abs_int + b_int);
    int32_t y_int = sign * (intermediate * intermediate + c_int);
    int32_t sigmoid_int = y_int / (1 << constant);

    gelu_in = gelu_in * (sigmoid_int + shift_int);

    return int8_t(gelu_in * M_stage3);

}

void linear_fused(int8_t *A, int8_t *B, int32_t *bias, int8_t *out, float M_gelu, float M_stage3)
{
    // compute fused gelu constants
    const float k = 1.4142;
    const int constant = 14;
    const float coef_0 = -0.2888;
    const float coef_1 = -1.769;
    const float coef_2 = 1 / coef_0;

    // int_erf
    float int_erf_scaling = M_gelu / k;
    int b_int = int(coef_1 / int_erf_scaling);
    int c_int = int(coef_2 / (int_erf_scaling * int_erf_scaling));
    float sigmoid_scaling_factor = int_erf_scaling * int_erf_scaling * coef_0;
    sigmoid_scaling_factor = sigmoid_scaling_factor * (1 << constant);

    int32_t shift_int = int32_t(1 / sigmoid_scaling_factor);

    // buffers for tile mmult
    int32_t out_block[TILE_SIZE][TILE_SIZE];
    int8_t B_line[TILE_SIZE];

    #pragma HLS array_partition dim=2 complete variable=out_block
    //#pragma HLS array_partition dim=1 type=cyclic factor=32 variable=out_block
    #pragma HLS array_partition dim=1 complete variable=B_line

    for (int it = 0; it < CFG::seqlen/TILE_SIZE; ++it)
    {
        for (int jt = 0; jt < CFG::ffdim/TILE_SIZE; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE; ++i){
                for (int j = 0; j < TILE_SIZE; ++j){
                    #pragma HLS unroll
                    out_block[i][j] = bias[jt*TILE_SIZE + j];
                }
            }

            for (int kt = 0; kt < CFG::dmodel/TILE_SIZE; ++kt)
            {
                for (int k = 0; k < TILE_SIZE; ++k)
                {
                    
                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE; ++j){
                        B_line[j] = B[(kt * TILE_SIZE + k) * CFG::ffdim + jt * TILE_SIZE + j];
                    }

                    for (int i = 0; i < TILE_SIZE; ++i){
                        //#pragma HLS unroll factor=4
                        #pragma HLS PIPELINE II=1
                        int8_t Ai = A[(it * TILE_SIZE + i) * CFG::dmodel + kt * TILE_SIZE + k];
                        for (int j = 0; j < TILE_SIZE; ++j){
                            #pragma HLS unroll complete
                            out_block[i][j] += Ai * B_line[j];
                            
                        }
                    }
                }
            }

            // apply gelu and write output
            for (int i = 0; i < TILE_SIZE; ++i){
                #pragma HLS PIPELINE II=1
                for (int j = 0; j < TILE_SIZE; ++j){
                    out[(it * TILE_SIZE + i) * CFG::ffdim + jt * TILE_SIZE + j] = gelu_fused(out_block[i][j], M_gelu, M_stage3, b_int, c_int, shift_int);
                }
            }
            
        }
        
    }
}

extern "C" {
void stage3(int8_t *fc_in, int8_t *dense_weight_t, int32_t *dense_bias, int8_t *dense_out, float dense_acc_scale, float M_stage3)
{
    /*
    High level: inputs are int8_t, output is int8_t. The linear layer goes from int8 -> int32. then, apply GeLU (which takes scaling_factor
    that is related to the int32_t quantization) which goes from int32 -> int32, then requantize2 to int8. This can all be fused.

    dense_acc_scale:    the scaling factor used within I-GeLU
    M_stage 3:          the requantization factor used to quantize the output of GeLU from 32 to 8 bits.
    */

    linear_fused(fc_in, dense_weight_t, dense_bias, dense_out, dense_acc_scale, M_stage3);
}
}
