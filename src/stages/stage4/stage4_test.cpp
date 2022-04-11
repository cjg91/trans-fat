#include "stage4.hpp"
#include "../pipeline.hpp"
#include <iostream>

void printmat(int8_t *A, const int M, const int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << int(A[i * N + j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void genmat(T *A, const int M, const int N, const int mod)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (i * N + j) % mod;
        }
    }
}

template <typename T>
const bool check(T *A, T *B, const int M, const int N)
{
    for (int i = 0; i < M * N; i++)
    {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

int main()
{

    int8_t* fc_in = new int8_t[CFG::seqlen * CFG::ffdim];
    int8_t* skip_conn = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t* dense_weight_t = new int8_t[CFG::ffdim * CFG::dmodel];
    int32_t* dense_bias = new int32_t[CFG::dmodel];
    int16_t* norm_weight = new int16_t[CFG::dmodel];
    int16_t* norm_bias = new int16_t[CFG::dmodel];

    genmat(fc_in, CFG::seqlen, CFG::ffdim, 7);
    genmat(skip_conn, CFG::seqlen, CFG::dmodel, 5);
    genmat(dense_weight_t, CFG::ffdim, CFG::dmodel, 9);
    genmat(dense_bias, CFG::dmodel, 1, 44);
    genmat(norm_weight, CFG::dmodel, 1, 23);
    genmat(norm_bias, CFG::dmodel, 1, 11);

    float M_residual = 2;
    float M_dense_acc = 0.04;
    float M_stage4 = 0.3;

    int8_t* dense_out_gt = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t *dense_out = new int8_t[CFG::seqlen * CFG::dmodel];

    stage4_gt(fc_in, skip_conn, M_residual, dense_weight_t, dense_bias, dense_out_gt, M_dense_acc, norm_weight, norm_bias, M_stage4);
    stage4_gt(fc_in, skip_conn, M_residual, dense_weight_t, dense_bias, dense_out, M_dense_acc, norm_weight, norm_bias, M_stage4);

    std::cout << "dense_out: " << (check(dense_out_gt, dense_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    delete [] fc_in;
    delete [] skip_conn;
    delete [] dense_weight_t;
    delete [] dense_bias;
    delete [] norm_weight;
    delete [] norm_bias;
    delete [] dense_out;
    delete [] dense_out_gt;

    return 0;
}