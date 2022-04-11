#pragma once

#include <cstdint>

void stage3_gt(int8_t* fc_in, int8_t* dense_weight_t, int32_t* dense_bias, int8_t* dense_out, float dense_acc_scale, float M_stage3);
void stage3(int8_t *fc_in, int8_t *dense_weight_t, int32_t *dense_bias, int8_t *dense_out, float dense_acc_scale, float M_stage3);