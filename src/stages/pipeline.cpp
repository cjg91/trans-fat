#include "pipeline.hpp"
#include "stage1/stage1.hpp"
#include "stage2/stage2.hpp"
#include "stage3/stage3.hpp"
#include "stage4/stage4.hpp"

void fpga2_gt(stage3_args_t s3_args, stage4_args_t s4_args)
{
    int8_t fc3_to_fc4_buff[CFG::seqlen][CFG::ffdim];
    int8_t *buff = &fc3_to_fc4_buff[0][0];

    stage3_gt(s3_args.fc_in, s3_args.dense_weight_t, s3_args.dense_bias, buff, s3_args.dense_acc_scale, s3_args.M_stage3);
    stage4_gt(buff, s4_args.skip_conn, s4_args.M_residual, s4_args.dense_weight_t, s4_args.dense_bias, s4_args.dense_out,
              s4_args.M_dense_acc, s4_args.norm_weight, s4_args.norm_bias, s4_args.M_stage4);
}

void fpga2(stage3_args_t s3_args, stage4_args_t s4_args)
{
    int8_t fc3_to_fc4_buff[CFG::seqlen][CFG::ffdim];
    int8_t *buff = &fc3_to_fc4_buff[0][0];

    stage3_gt(s3_args.fc_in, s3_args.dense_weight_t, s3_args.dense_bias, buff, s3_args.dense_acc_scale, s3_args.M_stage3);
    stage4_gt(buff, s4_args.skip_conn, s4_args.M_residual, s4_args.dense_weight_t, s4_args.dense_bias, s4_args.dense_out,
              s4_args.M_dense_acc, s4_args.norm_weight, s4_args.norm_bias, s4_args.M_stage4);
}