#include <cstdint>
#include "pipeline.hpp"

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

void requantize(int32_t* in, int8_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = int8_t(in[i*cols+j] * M_scale);
        }
    }
}

void stage1_gt(int8_t* in, int8_t* query_out, int8_t* key_out, int8_t* value_out, int8_t* query_weight_t, int32_t* query_bias, int8_t* key_weight_t, int32_t* key_bias, int8_t* value_weight_t, int32_t* value_bias, float M_query, float M_key, float M_value) {

    auto query = new int32_t[seqlen*dmodel];
    auto key = new int32_t[seqlen*dmodel];
    auto value = new int32_t[seqlen*dmodel];

    linear_sw(in, query_weight_t, query_bias, query, seqlen, dmodel, dmodel);
    linear_sw(in, key_weight_t, key_bias, key, seqlen, dmodel, dmodel);
    linear_sw(in, value_weight_t, value_bias, value, seqlen, dmodel, dmodel);

    requantize(query, query_out, seqlen, dmodel, M_query);
    requantize(key, key_out, seqlen, dmodel, M_key);
    requantize(value, value_out, seqlen, dmodel, M_value);

    delete[] query;
    delete[] key;
    delete[] value;
}