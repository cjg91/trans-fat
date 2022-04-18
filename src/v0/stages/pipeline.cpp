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

void fpga1(stage1_args_t s1_args, stage2_args_t s2_args)
{
    int8_t q[CFG::seqlen * CFG::dmodel];
    int8_t k[CFG::seqlen * CFG::dmodel];
    int8_t v[CFG::seqlen * CFG::dmodel];

    stage1(s1_args.in, q, k, v, s1_args.query_weight_t, s1_args.query_bias, s1_args.key_weight_t, s1_args.key_bias,
              s1_args.value_weight_t, s1_args.value_bias, s1_args.M_query, s1_args.M_key, s1_args.M_value);

    stage2(q, k, v, s1_args.in, s2_args.out, s2_args.dense_weight_t, s2_args.dense_bias, s2_args.M_attention_probs, s2_args.M_attention_out,
              s2_args.M_dense_out, s2_args.M_residual, s2_args.norm_weight, s2_args.norm_bias, s2_args.M_stage2);
    
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
