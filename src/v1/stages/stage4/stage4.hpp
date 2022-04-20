#pragma once

#include <sys/types.h>

const int TILE_SIZE4 = 128;

extern "C"
{
void stage4_gt(int8_t* fc_in, int8_t* skip_conn, float M_residual, int8_t* dense_weight_t, int32_t* dense_bias, int8_t* dense_out, float M_dense_acc, int16_t* norm_weight, int16_t* norm_bias, float M_stage4);
void stage4(int8_t *fc_in, int8_t *skip_conn, float M_residual, int8_t *dense_weight_t, int32_t *dense_bias, int8_t *dense_out, float M_dense_acc, int16_t *norm_weight, int16_t *norm_bias, float M_stage4);
}
