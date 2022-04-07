#include <cstdint>

void stage2(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, float scores_scale, float M_attention_probs, float M_attention_out, float M_dense_out, int16_t* norm_weight, int16_t* norm_bias, float M_stage2);