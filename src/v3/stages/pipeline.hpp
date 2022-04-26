#pragma once
#include <inttypes.h>

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
void fpga2_gt(stage3_args_t s3_args, stage4_args_t s4_args);

extern "C" {
void fpga1(int8_t *stage1_in, int8_t *query, int8_t *key, int8_t *value, int8_t *stage1_query_weight_t, int32_t *stage1_query_bias,
           int8_t *stage1_key_weight_t, int32_t *stage1_key_bias, int8_t *stage1_value_weight_t, int32_t *stage1_value_bias, float stage1_M_query, 
           float stage1_M_key, float stage1_M_value, int8_t *stage2_out, int8_t *stage2_dense_weight_t, int32_t *stage2_dense_bias, 
           float stage2_M_attention_probs, float stage2_M_attention_out, float stage2_M_dense_out, float stage2_M_residual, 
           int16_t *stage2_norm_weight, int16_t *stage2_norm_bias, float M_stage2);

void fpga2(int8_t *stage3_fc_in, int8_t* stage3_dense_weight_t, int32_t *stage3_dense_bias, float stage3_dense_acc_scale, float M_stage3, 
           int8_t *fc3_to_fc4_buff, int8_t *stage4_dense_weight_t, int8_t *stage4_dense_out, int32_t *stage4_dense_bias, int16_t* stage4_norm_weight,
           int16_t *stage4_norm_bias, float stage4_M_residual, float stage4_M_dense_acc, float M_stage4);
}


