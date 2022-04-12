#include "stage4.hpp"
#include <math.h>
#include <stdio.h>
#include "../../config.hpp"

void linear_sw(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out[i*M+j] = bias[j];
            for (int k = 0; k < K; k++) {
                out[i*M+j] += A[i*K+k] * B[k*M+j];
            }
        }
    }
}
template <typename in_t, typename out_t>
void requantize(in_t* in, out_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = out_t(in[i*cols+j] * M_scale);
        }
    }
}

void add_skip(int8_t* inout, const int8_t* skip_conn, const int32_t len)
{
    for (int i = 0; i < len; ++i){
        inout[i] += skip_conn[i];
    }
}

void mean(int16_t* act, int16_t* out)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        int32_t acc32 = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            acc32 += act[i * CFG::dmodel + j];
        }
        out[i] = int16_t(acc32/CFG::dmodel);
    }
}

void sum(int16_t* in, int16_t* out)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        out[i] = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            out[i] += in[i * CFG::dmodel + j];
        }
    }
}

void diff(int16_t* y, int16_t* act, int16_t* means)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        for (int j = 0; j < CFG::dmodel; ++j){  
            y[i*CFG::dmodel + j] = act[i*CFG::dmodel + j] - means[i];
        }
    }
}

void div(int16_t* y, int16_t* stdev)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        for (int j = 0; j < CFG::dmodel; ++j){
            y[i * CFG::dmodel + j] /= stdev[i];
        }
    }
}

void layernorm_sw(int16_t* act, int16_t* y, int16_t* norm_weight, int16_t* norm_bias, float scaling_factor)
{
    int16_t* means = new int16_t[CFG::seqlen];
    int16_t* y_sq = new int16_t[CFG::seqlen * CFG::dmodel];
    int16_t* var = new int16_t[CFG::seqlen];
    int16_t *stdev = new int16_t[CFG::seqlen];

    mean(act, means);
    diff(y, act, means);
    
    // square elements
    for (int i = 0; i < CFG::dmodel * CFG::seqlen; ++i){
        y_sq[i] = int16_t(int32_t((y[i] * y[i])) / CFG::dmodel);
    }
    
    // compute var by summing y^2
    sum(y_sq, var);
    
    // calculate constant for std computation
    int16_t C = int16_t(CFG::eps / scaling_factor);
    
    // compute std
    for (int i = 0; i < CFG::seqlen; ++i){
        stdev[i] = int16_t(sqrt(float(var[i] + C)));
    }

    // perform the division on each element in y
    div(y, stdev);
    
    // perform macs
    for (int i = 0; i < CFG::dmodel; ++i){
        for (int j = 0; j < CFG::seqlen; ++j){
            y[j * CFG::dmodel + i] = int16_t((y[j * CFG::dmodel + i] * norm_weight[i] + norm_bias[i]) * scaling_factor);
        }
    }

   delete [] means;
   delete [] y_sq;
   delete [] var;
   delete [] stdev;
    
}

void stage4_gt(int8_t *fc_in, int8_t *skip_conn, float M_residual, int8_t *dense_weight_t, int32_t *dense_bias, int8_t *dense_out, float M_dense_acc, int16_t *norm_weight, int16_t *norm_bias, float M_stage4)
{
    int32_t* fc_out = new int32_t[CFG::seqlen * CFG::dmodel];
    int8_t* fc_out_quant = new int8_t[CFG::seqlen * CFG::dmodel];
    int16_t* ln_fc_in = new int16_t[CFG::seqlen * CFG::dmodel];
    int16_t* ln_out = new int16_t[CFG::seqlen * CFG::dmodel];

    linear_sw(fc_in, dense_weight_t, dense_bias, fc_out, CFG::seqlen, CFG::dmodel, CFG::ffdim);
    requantize(fc_out, fc_out_quant, CFG::seqlen, CFG::dmodel, M_dense_acc);
    add_skip(fc_out_quant, skip_conn, CFG::seqlen * CFG::dmodel);
    requantize(fc_out_quant, ln_fc_in, CFG::seqlen, CFG::dmodel, M_residual);
    layernorm_sw(ln_fc_in, ln_out, norm_weight, norm_bias, M_residual);
    requantize(ln_out, dense_out, CFG::seqlen, CFG::dmodel, M_stage4);

    delete [] fc_out;
    delete [] fc_out_quant;
    delete [] ln_fc_in;
    delete [] ln_out;
}


/**** ^^ SW Ground Truth Above ^^ ****/

void linear_fused(int8_t* A, int8_t* B, int32_t* bias, int16_t* out, const int N, const int M, const int K, int8_t* skip_conn, float M_dense, float M_residual) {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int32_t acc32 = bias[j];
            for (int k = 0; k < K; k++) {
                acc32 += A[i*K+k] * B[k*M+j];
            }
            int8_t acc8 = int8_t(acc32 * M_dense) + skip_conn[i * M + j];
            out[i * M + j] = int16_t(acc8 * M_residual);
        }
    }
}

void layernorm_fused(int16_t *act, int8_t *out, int16_t *norm_weight, int16_t *norm_bias, float scaling_factor, float M_stage)
{
    // calculate constant for std computation
    const int16_t C = int16_t(CFG::eps / scaling_factor);

    // for some reason rn if I fuse this int the next loops it doesn't work
    // we'll want to probably break these up and pipeline anyway so leaving for now
    for (int i = 0; i < CFG::seqlen; ++i){
        int32_t macc = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            macc += act[i * CFG::dmodel + j];
        }
        int16_t m = int16_t(macc/CFG::dmodel);
        for (int j = 0; j < CFG::dmodel; ++j){  
            act[i*CFG::dmodel + j] = act[i*CFG::dmodel + j] - m;
        }
    } 

    for (int i = 0; i < CFG::seqlen; ++i){
        int16_t acc16 = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            acc16 += int16_t(act[i * CFG::dmodel + j]*int32_t(act[i * CFG::dmodel + j])/CFG::dmodel);
        }
        int16_t stdev = int16_t(sqrt(float(acc16 + C)));

        for (int j = 0; j < CFG::dmodel; ++j){
            act[i * CFG::dmodel + j] /= stdev;
            int16_t acc16 = int16_t((act[i * CFG::dmodel + j] * norm_weight[j] + norm_bias[j]) * scaling_factor);
            out[i * CFG::dmodel + j] = int8_t(acc16 * M_stage);
        }
    }

}

void stage4(int8_t *fc_in, int8_t *skip_conn, float M_residual, int8_t *dense_weight_t, int32_t *dense_bias, int8_t *dense_out, float M_dense_acc, int16_t *norm_weight, int16_t *norm_bias, float M_stage4)
{
    int16_t fc_ln_buff[CFG::seqlen][CFG::dmodel];
    int16_t* buff = &fc_ln_buff[0][0];

    linear_fused(fc_in, dense_weight_t, dense_bias, buff, CFG::seqlen, CFG::dmodel, CFG::ffdim, skip_conn, M_dense_acc, M_residual);
    layernorm_fused(buff, dense_out, norm_weight, norm_bias, M_residual, M_stage4);

}
