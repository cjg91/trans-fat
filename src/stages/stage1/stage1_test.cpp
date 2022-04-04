#include "stage1.hpp"
#include "../pipeline.hpp"
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

template<typename T>
const bool check(T* A, T* B, const int M, const int N)
{
    for (int i = 0; i < M*N; i++) {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

int main() {

    auto hidden_states = new int8_t [CFG::seqlen*CFG::dmodel];
    
    auto query_weight_t = new int8_t [CFG::dmodel*CFG::dmodel];
    auto key_weight_t = new int8_t [CFG::dmodel*CFG::dmodel];
    auto value_weight_t = new int8_t [CFG::dmodel*CFG::dmodel];
    
    auto query_bias = new int32_t [CFG::dmodel];    
    auto key_bias = new int32_t [CFG::dmodel];
    auto value_bias = new int32_t [CFG::dmodel];

    genmat(hidden_states, CFG::seqlen, CFG::dmodel, 7);

    genmat(query_weight_t, CFG::dmodel, CFG::dmodel, 9);
    genmat(key_weight_t, CFG::dmodel, CFG::dmodel, 11);
    genmat(value_weight_t, CFG::dmodel, CFG::dmodel, 13);

    genmat(query_bias, 1, CFG::dmodel, 63);
    genmat(key_bias, 1, CFG::dmodel, 65);
    genmat(value_bias, 1, CFG::dmodel, 67);


    float M_query = 0.5;
    float M_key = 0.4;
    float M_value = 0.3;

    auto query_out_gt = new int8_t [CFG::seqlen*CFG::dmodel];
    auto key_out_gt = new int8_t [CFG::seqlen*CFG::dmodel];
    auto value_out_gt = new int8_t [CFG::seqlen*CFG::dmodel];
    auto query_out = new int8_t[CFG::seqlen * CFG::dmodel];
    auto key_out = new int8_t[CFG::seqlen * CFG::dmodel];
    auto value_out = new int8_t[CFG::seqlen * CFG::dmodel];

    stage1_gt(hidden_states, query_out_gt, key_out_gt, value_out_gt, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias, M_query, M_key, M_value);
    stage1(hidden_states, query_out, key_out, value_out, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias, M_query, M_key, M_value);

    std::cout << "query_out: " << (check(query_out_gt, query_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    std::cout << "key_out:   " << (check(key_out_gt, key_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    std::cout << "value_out: " << (check(value_out_gt, value_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    return 0;
}