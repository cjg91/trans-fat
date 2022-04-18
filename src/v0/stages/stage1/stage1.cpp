#include <sys/types.h>
#include <iostream>
#include "config.hpp"

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
void linear_fused(int8_t* A, int8_t* B, int32_t* bias, int8_t* out, const int N, const int M, const int K, const float M_scale) {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // Initialize accumulator
            int32_t acc32 = bias[j];
            for (int k = 0; k < K; k++) {
                acc32 += int32_t(A[i*K+k] * B[k*M+j]);
            }
            out[i * M + j] = int8_t(acc32 * M_scale);
        }
    }
}

extern "C" {
void stage1(int8_t *in, int8_t *query_out, int8_t *key_out, int8_t *value_out, int8_t *query_weight_t, int32_t *query_bias, int8_t *key_weight_t, int32_t *key_bias, int8_t *value_weight_t, int32_t *value_bias, float M_query, float M_key, float M_value)
{
    linear_fused(in, query_weight_t, query_bias, query_out, CFG::seqlen, CFG::dmodel, CFG::dmodel, M_query);
    linear_fused(in, key_weight_t, key_bias, key_out, CFG::seqlen, CFG::dmodel, CFG::dmodel, M_key);
    linear_fused(in, value_weight_t, value_bias, value_out, CFG::seqlen, CFG::dmodel, CFG::dmodel, M_value);
}
}
