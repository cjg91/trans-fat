#include <cstdint>
void gelu_sw(int32_t* gelu_in, int32_t* gelu_out, int rows, int cols, float scaling_factor);
void stage3_gt(int8_t* fc_in, int8_t* dense_weight_t, int32_t* dense_bias, int8_t* dense_out, float dense_acc_scale, float M_stage3);