#include "pipeline.hpp"
#include "../config.hpp"
#include <memory.h>
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

    /********** STAGE 1 ARGS ***********/
    stage1_args_t s1_args;
    s1_args.in = new int8_t[CFG::seqlen * CFG::dmodel];
    s1_args.query_weight_t = new int8_t[CFG::dmodel * CFG::dmodel];
    s1_args.key_weight_t = new int8_t[CFG::dmodel * CFG::dmodel];
    s1_args.value_weight_t = new int8_t[CFG::dmodel * CFG::dmodel];
    s1_args.query_bias = new int32_t[CFG::dmodel];
    s1_args.key_bias = new int32_t[CFG::dmodel];
    s1_args.value_bias = new int32_t[CFG::dmodel];
    s1_args.M_query = 0.5;
    s1_args.M_key = 0.4;
    s1_args.M_value = 0.3;

    auto query = new int8_t[CFG::seqlen * CFG::dmodel];
    auto key = new int8_t[CFG::seqlen * CFG::dmodel];
    auto value = new int8_t[CFG::seqlen * CFG::dmodel];

    genmat(s1_args.in, CFG::seqlen, CFG::dmodel, 7);
    genmat(s1_args.query_weight_t, CFG::dmodel, CFG::dmodel, 9);
    genmat(s1_args.key_weight_t, CFG::dmodel, CFG::dmodel, 11);
    genmat(s1_args.value_weight_t, CFG::dmodel, CFG::dmodel, 13);
    genmat(s1_args.query_bias, 1, CFG::dmodel, 63);
    genmat(s1_args.key_bias, 1, CFG::dmodel, 65);
    genmat(s1_args.value_bias, 1, CFG::dmodel, 67);

    /********** STAGE 2 ARGS ***********/
    stage2_args_t s2_args;
    s2_args.out = new int8_t[CFG::seqlen * CFG::dmodel];
    s2_args.dense_weight_t = new int8_t[CFG::dmodel*CFG::dmodel];
    s2_args.dense_bias = new int32_t[CFG::dmodel];
    s2_args.norm_weight = new int16_t[CFG::dmodel];
    s2_args.norm_bias = new int16_t[CFG::dmodel];
    s2_args.M_attention_probs = 100;
    s2_args.M_attention_out = 0.1;
    s2_args.M_dense_out = 0.1;
    s2_args.M_residual = 1;
    s2_args.M_stage2 = 1;

    genmat(s2_args.dense_weight_t, CFG::dmodel, CFG::dmodel, 13);
    genmat(s2_args.dense_bias, CFG::dmodel, 1, 61);
    genmat(s2_args.norm_weight, CFG::dmodel, 1, 62);
    genmat(s2_args.norm_bias, CFG::dmodel, 1, 69);

    /********** STAGE 3 ARGS ***********/
    stage3_args_t s3_args;
    s3_args.fc_in = s2_args.out;
    s3_args.dense_weight_t = new int8_t[CFG::dmodel * CFG::ffdim];
    s3_args.dense_bias = new int32_t[CFG::ffdim];
    s3_args.dense_acc_scale = 0.004;
    s3_args.M_stage3 = 0.3;

    auto fc3_to_fc4_buff = new int8_t[CFG::seqlen * CFG::ffdim];

    genmat(s3_args.dense_weight_t, CFG::dmodel, CFG::ffdim, 13);
    genmat(s3_args. dense_bias, 1, CFG::ffdim, 71);

    /********** STAGE 4 ARGS ***********/
    stage4_args_t s4_args;
    s4_args.skip_conn = s3_args.fc_in;
    s4_args.dense_weight_t = new int8_t[CFG::ffdim * CFG::dmodel];
    s4_args.norm_bias = new int16_t[CFG::dmodel];
    s4_args.norm_weight = new int16_t[CFG::dmodel];
    s4_args.dense_bias = new int32_t[CFG::dmodel];
    s4_args.dense_out = new int8_t[CFG::seqlen * CFG::dmodel];
    s4_args.M_residual = 2;
    s4_args.M_dense_acc = 1;
    s4_args.M_stage4 = 1;

    genmat(s4_args.dense_weight_t, CFG::ffdim, CFG::dmodel, 9);
    genmat(s4_args.dense_bias, CFG::dmodel, 1, 44);
    genmat(s4_args.norm_weight, CFG::dmodel, 1, 23);
    genmat(s4_args.norm_bias, CFG::dmodel, 1, 11);

    /*********************** run fpga layer *********************/
    fpga1(s1_args.in, query, key, value, s1_args.query_weight_t, s1_args.query_bias, s1_args.key_weight_t, s1_args.key_bias, s1_args.value_weight_t,
          s1_args.value_bias, s1_args.M_query, s1_args.M_key, s1_args.M_value, s2_args.out, s2_args.dense_weight_t, s2_args.dense_bias, 
          s2_args.M_attention_probs, s2_args.M_attention_out, s2_args.M_dense_out, s2_args.M_residual, 
          s2_args.norm_weight, s2_args.norm_bias, s2_args.M_stage2);

    fpga2(s3_args.fc_in, s3_args.dense_weight_t, s3_args.dense_bias, s3_args.dense_acc_scale, s3_args.M_stage3,
          fc3_to_fc4_buff, s4_args.dense_weight_t, s4_args.dense_out, s4_args.dense_bias, s4_args.norm_weight, s4_args.norm_bias,
          s4_args.M_residual, s4_args.M_dense_acc, s4_args.M_stage4);

    // save layer output for comparison
    int8_t *test_out = new int8_t[CFG::seqlen * CFG::dmodel];
    memcpy(test_out, s4_args.dense_out, CFG::seqlen * CFG::dmodel * sizeof(int8_t));

    /****************** run ground truth layer ******************/
    fpga1_gt(s1_args, s2_args);
    fpga2_gt(s3_args, s4_args);

    std::cout << "dense_out: " << (check(s4_args.dense_out, test_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    // memory cleanup
    delete [] s1_args.in;
    delete [] s1_args.query_weight_t;
    delete [] s1_args.key_weight_t;
    delete [] s1_args.value_weight_t;
    delete [] s1_args.query_bias;
    delete [] s1_args.key_bias;
    delete [] s1_args.value_bias;
    delete [] query;
    delete [] key;
    delete [] value;
    delete [] s2_args.out;
    delete [] s2_args.norm_weight;
    delete [] s2_args.norm_bias;
    delete [] s3_args.dense_weight_t;
    delete [] s3_args.dense_bias;
    delete [] fc3_to_fc4_buff;
    delete [] s4_args.dense_weight_t;
    delete [] s4_args.norm_bias;
    delete [] s4_args.norm_weight;
    delete [] s4_args.dense_bias;
    delete [] s4_args.dense_out;

    return EXIT_SUCCESS;
}
