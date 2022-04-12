#pragma once
#include <cstdint>

namespace CFG
{
    constexpr int seqlen = 128;
    constexpr int nhead = 12;
    constexpr int dhead = 64;
    constexpr int dmodel = 768;
    constexpr int ffdim = 3072;
    constexpr float eps = 1e-5;
} // namespace CONFIG

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

void fpga2_gt(stage3_args_t s3_args, stage4_args_t s4_args);
void fpga2(stage3_args_t s3_args, stage4_args_t s4_args);