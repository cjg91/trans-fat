#include <inttypes.h>
#include <iostream>
#include "config.hpp"
#include "stage1.hpp"
/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/

void linear_sw1(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {
    
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

void requantize1(int32_t* in, int8_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = int8_t(in[i*cols+j] * M_scale);
        }
    }
}

extern "C" {
void stage1_gt(int8_t* in, int8_t* query_out, int8_t* key_out, int8_t* value_out, int8_t* query_weight_t, int32_t* query_bias, int8_t* key_weight_t, int32_t* key_bias, int8_t* value_weight_t, int32_t* value_bias, float M_query, float M_key, float M_value) {

    auto query = new int32_t[CFG::seqlen*CFG::dmodel];
    auto key = new int32_t[CFG::seqlen*CFG::dmodel];
    auto value = new int32_t[CFG::seqlen*CFG::dmodel];

    linear_sw1(in, query_weight_t, query_bias, query, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw1(in, key_weight_t, key_bias, key, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw1(in, value_weight_t, value_bias, value, CFG::seqlen, CFG::dmodel, CFG::dmodel);

    requantize1(query, query_out, CFG::seqlen, CFG::dmodel, M_query);
    requantize1(key, key_out, CFG::seqlen, CFG::dmodel, M_key);
    requantize1(value, value_out, CFG::seqlen, CFG::dmodel, M_value);

    delete [] query;
    delete [] key;
    delete [] value;
}
}
/*^^^^^^^^^^^^^^^^^^^ END GT ^^^^^^^^^^^^^^^^^^^*/

/****************** Stage Kernel Code *********************/

/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/
void linear_fused(int8_t* A_T, int8_t* B, int32_t* bias, int8_t* out, const float M_scale) 
{
    // buffers for tile mmult
    int32_t out_block[TILE_SIZE1][TILE_SIZE1];
    int8_t B_line[TILE_SIZE1];
    int8_t A_line[TILE_SIZE1];

    #pragma HLS array_partition dim=2 complete variable=out_block
    #pragma HLS array_partition dim=1 complete variable=B_line

    for (int it = 0; it < CFG::seqlen/TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel/TILE_SIZE1; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE1; ++i){
                for (int j = 0; j < TILE_SIZE1; ++j){
                    out_block[i][j] = bias[jt*TILE_SIZE1 + j];
                }
            }

            for (int kt = 0; kt < CFG::dmodel/TILE_SIZE1; ++kt)
            {
                for (int k = 0; k < TILE_SIZE1; ++k)
                {
                    
                    // read B values into vector
                    for (int j = 0; j < TILE_SIZE1; ++j){
                        B_line[j] = B[(kt * TILE_SIZE1 + k) * CFG::dmodel + jt * TILE_SIZE1 + j];
                    }
                    for (int i = 0; i < TILE_SIZE1; ++i) {
                        A_line[i] = A_T[(kt * TILE_SIZE1 + k) * CFG::seqlen + it * TILE_SIZE1 + i];
                    }

                    for (int i = 0; i < TILE_SIZE1; ++i){
                        #pragma HLS PIPELINE II=1
                        int8_t Ai = A_line[i];
                        for (int j = 0; j < TILE_SIZE1; ++j){
                            #pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                            
                        }
                    }
                }
            }

            for (int i = 0; i < TILE_SIZE1; ++i){
                #pragma HLS PIPELINE II=1
                for (int j = 0; j < TILE_SIZE1; ++j){
                    out[(it * TILE_SIZE1 + i) * CFG::dmodel + jt * TILE_SIZE1 + j] = int8_t(out_block[i][j] * M_scale);
                }
            }
            
        }
        
    }
}

extern "C" {
void stage1(int8_t *in, int8_t *query_out, int8_t *key_out, int8_t *value_out, int8_t *query_weight_t, int32_t *query_bias, int8_t *key_weight_t, int32_t *key_bias, int8_t *value_weight_t, int32_t *value_bias, float M_query, float M_key, float M_value)
{   
    linear_fused(in, query_weight_t, query_bias, query_out, M_query);
    linear_fused(in, key_weight_t, key_bias, key_out, M_key);
    linear_fused(in, value_weight_t, value_bias, value_out, M_value);
}
}
