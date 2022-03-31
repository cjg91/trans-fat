#include "stage1.hpp"
#include "pipeline.hpp"
#include <iostream>

void printmat(int8_t* A, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << int(A[i*N+j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
void genmat(T* A, const int M, const int N, const int mod) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N+j] = (i*N+j) % mod;
        }
    }
}

int main() {

    auto hidden_states = new int8_t [seqlen*dmodel];
    
    auto query_weight_t = new int8_t [dmodel*dmodel];
    auto key_weight_t = new int8_t [dmodel*dmodel];
    auto value_weight_t = new int8_t [dmodel*dmodel];
    
    auto query_bias = new int32_t [dmodel];    
    auto key_bias = new int32_t [dmodel];
    auto value_bias = new int32_t [dmodel];

    genmat(hidden_states, seqlen, dmodel, 7);

    genmat(query_weight_t, dmodel, dmodel, 9);
    genmat(key_weight_t, dmodel, dmodel, 11);
    genmat(value_weight_t, dmodel, dmodel, 13);

    genmat(query_bias, 1, dmodel, 63);
    genmat(key_bias, 1, dmodel, 65);
    genmat(value_bias, 1, dmodel, 67);


    float M_query = 0.5;
    float M_key = 0.4;
    float M_value = 0.3;

    auto query_out = new int8_t [seqlen*dmodel];
    auto key_out = new int8_t [seqlen*dmodel];
    auto value_out = new int8_t [seqlen*dmodel];

    stage1_gt(hidden_states, query_out, key_out, value_out, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias, M_query, M_key, M_value);

    std::cout << "query_out" << std::endl;
    printmat(query_out, seqlen, dmodel);

    std::cout << "key_out" << std::endl;
    printmat(key_out, seqlen, dmodel);
    
    std::cout << "value_out" << std::endl;
    printmat(value_out, seqlen, dmodel);

    return 0;
}