#pragma once
#include <inttypes.h>

const int TILE_SIZE1 = 128;

extern "C"
{
void stage1_gt(int8_t* in, int8_t* query_out, int8_t* key_out, int8_t* value_out, int8_t* query_weight_t, int32_t* query_bias, int8_t* key_weight_t, int32_t* key_bias, int8_t* value_weight_t, int32_t* value_bias, float M_query, float M_key, float M_value);
void stage1(int8_t *in, int8_t *query_out, int8_t *key_out, int8_t *value_out, int8_t *query_weight_t, int32_t *query_bias, int8_t *key_weight_t, int32_t *key_bias, int8_t *value_weight_t, int32_t *value_bias, float M_query, float M_key, float M_value);
}
