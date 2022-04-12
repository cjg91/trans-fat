#include "stage1/stage1.hpp"
#include "stage2/stage2.hpp"
#include "stage3/stage3.hpp"
#include "stage4/stage4.hpp"
#include "../config.hpp"
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


void fpga1(stage1_args_t s1_args, stage2_args_t s2_args)
{
    int8_t q[CFG::seqlen * CFG::dmodel];
    int8_t k[CFG::seqlen * CFG::dmodel];
    int8_t v[CFG::seqlen * CFG::dmodel];

    stage1(s1_args.in, q, k, v, s1_args.query_weight_t, s1_args.query_bias, s1_args.key_weight_t, s1_args.key_bias,
              s1_args.value_weight_t, s1_args.value_bias, s1_args.M_query, s1_args.M_key, s1_args.M_value);

    stage2_gt(q, k, v, s1_args.in, s2_args.out, s2_args.dense_weight_t, s2_args.dense_bias, s2_args.M_attention_probs, s2_args.M_attention_out,
              s2_args.M_dense_out, s2_args.M_residual, s2_args.norm_weight, s2_args.norm_bias, s2_args.M_stage2);
    
}

void fpga2_gt(stage3_args_t s3_args, stage4_args_t s4_args)
{
    int8_t *fc3_to_fc4_buff = new int8_t[CFG::seqlen * CFG::ffdim];
    
    stage3_gt(s3_args.fc_in, s3_args.dense_weight_t, s3_args.dense_bias, fc3_to_fc4_buff, s3_args.dense_acc_scale, s3_args.M_stage3);
    
    stage4_gt(fc3_to_fc4_buff, s4_args.skip_conn, s4_args.M_residual, s4_args.dense_weight_t, s4_args.dense_bias, s4_args.dense_out,
              s4_args.M_dense_acc, s4_args.norm_weight, s4_args.norm_bias, s4_args.M_stage4);
              
    delete [] fc3_to_fc4_buff;
}

void fpga2(stage3_args_t s3_args, stage4_args_t s4_args)
{
    int8_t fc3_to_fc4_buff[CFG::seqlen][CFG::ffdim];
    int8_t *buff = &fc3_to_fc4_buff[0][0];

    stage3(s3_args.fc_in, s3_args.dense_weight_t, s3_args.dense_bias, buff, s3_args.dense_acc_scale, s3_args.M_stage3);
    stage4(buff, s4_args.skip_conn, s4_args.M_residual, s4_args.dense_weight_t, s4_args.dense_bias, s4_args.dense_out,
              s4_args.M_dense_acc, s4_args.norm_weight, s4_args.norm_bias, s4_args.M_stage4);
}