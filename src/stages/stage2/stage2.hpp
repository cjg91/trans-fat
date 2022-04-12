#include <cstdint>
void attention_scores(int8_t* query, int8_t* key, int32_t* out, const int seqlen, const int nhead, const int dhead);
void attention_values(int8_t* probs, int8_t* value, int32_t* attn_out, const int seqlen, const int nhead, const int dhead);

void stage2_gt(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, int8_t* dense_weight_t, int32_t* dense_bias, float M_attention_probs, float M_attention_out, float M_dense_out, float M_residual, int16_t* norm_weight, int16_t* norm_bias, float M_stage2);
