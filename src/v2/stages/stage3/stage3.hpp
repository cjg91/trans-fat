#pragma once

#include <inttypes.h>

const int TILE_SIZE = 128;
const int TILE_SIZE_J = 512;

extern "C"
{
void stage3_gt(int8_t* fc_in, int8_t* dense_weight_t, int32_t* dense_bias, int8_t* dense_out, float dense_acc_scale, float M_stage3);
void stage3(int8_t *fc_in, int8_t *dense_weight_t, int32_t *dense_bias, int8_t *dense_out, float dense_acc_scale, float M_stage3);
}
