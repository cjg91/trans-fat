#include "stage1.hpp"
#include "stage2.hpp"
#include "stage3.hpp"
#include "stage4.hpp"
#include "config.hpp"
#include "pipeline.hpp"
#include <stdio.h>


void fpga1_gt(stage1_args_t s1_args, stage2_args_t s2_args)
{
    int8_t *q = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t *k = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t *v = new int8_t[CFG::seqlen * CFG::dmodel];

    

    stage1_gt(s1_args.in, q, k, v, s1_args.query_weight_t, s1_args.query_bias, s1_args.key_weight_t, s1_args.key_bias,
              s1_args.value_weight_t, s1_args.value_bias, s1_args.M_query, s1_args.M_key, s1_args.M_value);

    stage2_gt(q, k, v, s1_args.in, s2_args.out, s2_args.dense_weight_t, s2_args.dense_bias, s2_args.M_attention_probs, s2_args.M_attention_out,
              s2_args.M_dense_out, s2_args.M_residual, s2_args.norm_weight, s2_args.norm_bias, s2_args.M_stage2);
    
    delete [] q;
    delete [] k;
    delete [] v;
}



void fpga2_gt(stage3_args_t s3_args, stage4_args_t s4_args)
{
    int8_t *fc3_to_fc4_buff = new int8_t[CFG::seqlen * CFG::ffdim];
    
    stage3_gt(s3_args.fc_in, s3_args.dense_weight_t, s3_args.dense_bias, fc3_to_fc4_buff, s3_args.dense_acc_scale, s3_args.M_stage3);
    
    stage4_gt(fc3_to_fc4_buff, s4_args.skip_conn, s4_args.M_residual, s4_args.dense_weight_t, s4_args.dense_bias, s4_args.dense_out,
              s4_args.M_dense_acc, s4_args.norm_weight, s4_args.norm_bias, s4_args.M_stage4);
              
    delete [] fc3_to_fc4_buff;
}

extern "C" {

void fpga1(int8_t *stage1_in, int8_t *query, int8_t *key, int8_t *value, int8_t *stage1_query_weight_t, int32_t *stage1_query_bias,
           int8_t *stage1_key_weight_t, int32_t *stage1_key_bias, int8_t *stage1_value_weight_t, int32_t *stage1_value_bias, float stage1_M_query, 
           float stage1_M_key, float stage1_M_value, int8_t *stage2_out, int8_t *stage2_dense_weight_t, int32_t *stage2_dense_bias, 
           float stage2_M_attention_probs, float stage2_M_attention_out, float stage2_M_dense_out, float stage2_M_residual, 
           int16_t *stage2_norm_weight, int16_t *stage2_norm_bias, float M_stage2)
{
    //#pragma HLS interface m_axi port=stage1_in bundle=gmem0
    //#pragma HLS interface m_axi port=query bundle=gmem0
    //#pragma HLS interface m_axi port=key bundle=gmem1
    //#pragma HLS interface m_axi port=value bundle=gmem2
    //#pragma HLS interface m_axi port=stage1_query_weight_t bundle=gmem0
    //#pragma HLS interface m_axi port=stage1_query_bias bundle=gmem0
    //#pragma HLS interface m_axi port=stage1_key_weight_t bundle=gmem1
    //#pragma HLS interface m_axi port=stage1_key_bias bundle=gmem1
    //#pragma HLS interface m_axi port=stage1_value_weight_t bundle=gmem2
    //#pragma HLS interface m_axi port=stage1_value_bias bundle=gmem2
    //#pragma HLS interface s_axilite port=stage1_M_query bundle=control
    //#pragma HLS interface s_axilite port=stage1_M_key bundle=control
    //#pragma HLS interface s_axilite port=stage1_M_value bundle=control
    //#pragma HLS interface m_axi port=stage2_out bundle=gmem1
    //#pragma HLS interface m_axi port=stage2_dense_weight_t bundle=gmem2
    //#pragma HLS interface m_axi port=stage2_dense_bias bundle=gmem1
    //#pragma HLS interface s_axilite port=stage2_M_attention_probs bundle=control
    //#pragma HLS interface s_axilite port=stage2_M_attention_out bundle=control
    //#pragma HLS interface s_axilite port=stage2_M_dense_out bundle=control
    //#pragma HLS interface s_axilite port=stage2_M_residual bundle=control
    //#pragma HLS interface m_axi port=stage2_norm_weight bundle=gmem0
    //#pragma HLS interface m_axi port=stage2_norm_bias bundle=gmem2
    //#pragma HLS interface s_axilite port=M_stage2 bundle=control
    //#pragma HLS interface s_axilite port=return bundle=control

    stage1(stage1_in, query, key, value, stage1_query_weight_t, stage1_query_bias, stage1_key_weight_t, stage1_key_bias,
              stage1_value_weight_t, stage1_value_bias, stage1_M_query, stage1_M_key, stage1_M_value);

    stage2(query, key, value, stage1_in, stage2_out, stage2_dense_weight_t, stage2_dense_bias, stage2_M_attention_probs, stage2_M_attention_out,
              stage2_M_dense_out, stage2_M_residual, stage2_norm_weight, stage2_norm_bias, M_stage2);
    
}

void fpga2(int8_t *stage3_fc_in, int8_t* stage3_dense_weight_t, int32_t *stage3_dense_bias, float stage3_dense_acc_scale, float M_stage3, 
           int8_t *fc3_to_fc4_buff, int8_t *stage4_dense_weight_t, int8_t *stage4_dense_out, int32_t *stage4_dense_bias, int16_t* stage4_norm_weight,
           int16_t *stage4_norm_bias, float stage4_M_residual, float stage4_M_dense_acc, float M_stage4)
{

    stage3(stage3_fc_in, stage3_dense_weight_t, stage3_dense_bias, fc3_to_fc4_buff, stage3_dense_acc_scale, M_stage3);
    stage4(fc3_to_fc4_buff, stage3_fc_in, stage4_M_residual, stage4_dense_weight_t, stage4_dense_bias, stage4_dense_out,
              stage4_M_dense_acc, stage4_norm_weight, stage4_norm_bias, M_stage4);
}
}
