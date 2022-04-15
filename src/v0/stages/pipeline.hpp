#pragma once
#include <cstdint>
extern "C"
{
typedef struct stage1_args_t
{
    int8_t *in, *query_weight_t, *key_weight_t, *value_weight_t;
    int32_t *query_bias, *key_bias, *value_bias;
    float M_query, M_key, M_value;
} stage1_args_t;

typedef struct stage2_args_t
{
    int8_t *out, *dense_weight_t;
    int16_t *norm_weight, *norm_bias;
    int32_t *dense_bias;
    float M_attention_probs, M_attention_out, M_dense_out, M_residual, M_stage2;
} stage2_args_t;

typedef struct stage3_args_t
{
    int8_t *fc_in, *dense_weight_t;
    int32_t *dense_bias;
    float dense_acc_scale, M_stage3;
} stage3_args_t;

typedef struct stage4_args_t
{
    int8_t* dense_weight_t, *skip_conn, *dense_out;
    int32_t* dense_bias;
    int16_t* norm_weight, *norm_bias;
    float M_residual, M_dense_acc, M_stage4;
} stage4_args_t;

void fpga1_gt(stage1_args_t s1_args, stage2_args_t s2_args);
void fpga1(stage1_args_t s1_args, stage2_args_t s2_args);

void fpga2_gt(stage3_args_t s3_args, stage4_args_t s4_args);
void fpga2(stage3_args_t s3_args, stage4_args_t s4_args);
}
