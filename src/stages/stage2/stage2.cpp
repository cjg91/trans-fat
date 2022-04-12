#include <cstdint>
#include <iostream>
#include <cmath>
#include "../../config.hpp"

/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/
void linear_sw2(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // Initialize accumulator
            out[i*M+j] = bias[j];
            for (int k = 0; k < K; k++) {
                out[i*M+j] += A[i*K+k] * B[k*M+j];
            }
        }
    }
}

template <typename in_t, typename out_t>
void requantize2(in_t* in, out_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = out_t(in[i*cols+j] * M_scale);
        }
    }
}

void scale (int32_t* y)
{
    int32_t divisor = std::sqrt(CFG::dmodel);
    for (int i = 0; i < CFG::nhead*CFG::seqlen*CFG::seqlen; ++i){
        y[i] /= divisor;
    }
}

void softmax(int32_t* in, int32_t* out) {
    // TODO: implement for real.. This just copies

    for (int n = 0; n < CFG::nhead; n++) {
        for (int i = 0; i < CFG::seqlen; i++) {
            for (int j = 0; j < CFG::seqlen; j++) {
                out[n*CFG::seqlen*CFG::seqlen + i*CFG::seqlen + j] = in[n*CFG::seqlen*CFG::seqlen + i*CFG::seqlen + j];
            }
        }
    }
}

void softmax(int32_t*in) {
    // another stub for softmax, used in fused op. operates in-place
}

void add_skip2(int8_t* inout, const int8_t* skip_conn, const int32_t len)
{
    for (int i = 0; i < len; ++i){
        inout[i] += skip_conn[i];
    }
}

void mean2(int16_t* act, int16_t* out)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        int32_t acc32 = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            acc32 += act[i * CFG::dmodel + j];
        }
        out[i] = int16_t(acc32/CFG::dmodel);
    }
}

void sum2(int16_t* in, int16_t* out)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        out[i] = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            out[i] += in[i * CFG::dmodel + j];
        }
    }
}

void diff2(int16_t* y, int16_t* act, int16_t* means)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        for (int j = 0; j < CFG::dmodel; ++j){  
            y[i*CFG::dmodel + j] = act[i*CFG::dmodel + j] - means[i];
        }
    }
}

void div2(int16_t* y, int16_t* stdev)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        for (int j = 0; j < CFG::dmodel; ++j){
            y[i * CFG::dmodel + j] /= stdev[i];
        }
    }
}

void layernorm_sw2(int16_t* act, int16_t* y, int16_t* norm_weight, int16_t* norm_bias, float scaling_factor)
{
    int16_t* means = new int16_t[CFG::seqlen];
    int16_t* y_sq = new int16_t[CFG::seqlen * CFG::dmodel];
    int16_t* var = new int16_t[CFG::seqlen];
    int16_t *stdev = new int16_t[CFG::seqlen];

    mean2(act, means);
    diff2(y, act, means);
    
    // square elements
    for (int i = 0; i < CFG::dmodel * CFG::seqlen; ++i){
        y_sq[i] = int16_t(int32_t((y[i] * y[i])) / CFG::dmodel);
    }
    
    // compute var by summing y^2
    sum2(y_sq, var);
    
    // calculate constant for std computation
    int16_t C = int16_t(CFG::eps / scaling_factor);
    
    // compute std
    for (int i = 0; i < CFG::seqlen; ++i){
        stdev[i] = int16_t(sqrt(float(var[i] + C)));
    }

    // perform the division on each element in y
    div2(y, stdev);
    
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
    

/**
 * 
 * What is transpose?
 * A[i][j] = A[i*COLS + j]
 * A_T[i][j] = A[j][i] = A[j*COLS+i]
 * 
 * To obtain indexing into original array from transpose array:
 *  - for each dimension in the transpose array, find the corresponding dimension in the original
 *  - index into this transpose dimension in the place where the original dimension was
 * 
 * This means:
 * if A[i][j][k] was transposed in a way that rotated the dimensions around the right, so it now looks like B[k][i][j],
 * the way you index B[i][j][k] is with A[j][k][i]. This makes the most sense if each dimension means something to you,
 * like nhead, seqlen, and dhead. 
 * 
 * Then, you can flatten you new reordered index by, for each index, multiplying it by the product of the dimensions
 * following it and adding that to your accumulated index.
*/

void attention_scores(int8_t* query, int8_t* key, int32_t* out, const int seqlen, const int nhead, const int dhead) {
    /*
    * query :   <seqlen, dmodel> -> <nhead, seqlen, dhead>
    * key:      <seqlen, dmodel> -> <nhead, dhead, seqlen>
    * out:      <nhead, seqlen, seqlen>
    * 
    * query reshape: view as <seqlen, nhead, dhead> (means changing bounds and adding another dimension)
    * query_reshape[i][j][k] = query[i*nhead*dhead + j*dhead + k]
    * query transpose to get <nhead, seqlen, dhead> switches i and j.
    * query_transpose[i][j][k] = query_reshape[j][i][k] = query[j*nhead*dhead +i*dhead + k]
    * 
    * key reshape: view as <seqlen, nhead, dhead>
    * key_reshape[i][j][k] = key[i*nhead*dhead + j*dhead + k]
    * key transpose to get to <nhead, dhead, seqlen>. go from (i,j,k) to (j,k,i)
    * key_transpose[i][j][k] = key_reshape[k][i][j] = key[k*nhead*dhead + i*dhead + j]
    * 
    * 
    * query_transpose[i1][i2][i3] = query[i2*nhead*dhead +i1*dhead + i3]
    * key_transpose[i1][i2][i3] = key[i3*nhead*dhead + i1*dhead + i2]
    */

   for (int n = 0; n < nhead; n++) {
       // compute matmul NHEAD times
       for (int i = 0; i < seqlen; i++) {
           for (int j = 0; j < seqlen; j++) {
                int32_t accum = 0;
                for (int k = 0; k < dhead; k++) {
                        // accum += query[n,i,k] * key[n, k, j]
                        accum += query[i*nhead*dhead +n*dhead + k] * key[j*nhead*dhead + n*dhead + k];
                }
                // out[n,i,j] = accum
                out[n*seqlen*seqlen + i*seqlen + j] = accum;
           }
       }
   }
}

void attention_values(int8_t* probs, int8_t* value, int32_t* attn_out, const int seqlen, const int nhead, const int dhead) {

    /**
     * probs: <nhead, seqlen, seqlen>
     * value: <seqlen, dmodel> -> <nhead, seqlen, dhead> (same reshape/transpose as query)
     * 
     * attn_out: <nhead, seqlen, dhead> -> <seqlen, dmodel> (how do you index to do this in one shot)
     * attn_out[i1][i2][i3] = attn_out[i1*seqlen*dhead + i2*dhead + i3]
     * att_out_transpose is <seqlen, nhead, dhead>
     * att_out_transpose[i1][i2][i3] = attn_out[i2][i1][i3] = attn_out[i2*nhead*dhead + i1*dhead + i3]
     * 
     * value_transpose[i1][i2][i3] = value[i2*nhead*dhead +i1*dhead + i3]
     * 
    */

   for (int n = 0; n < nhead; n++) {
       for (int i = 0; i < seqlen; i++) {
           for (int j = 0; j < dhead; j++) {
               int32_t accum = 0;
               for (int k = 0; k < seqlen; k++) {
                   // attn_out[n][i][j] += probs[n][i][k] * value[n][k][j]
                    accum += probs[n*seqlen*seqlen + i*seqlen + k] * value[k*nhead*dhead +n*dhead + j];
               }
               // writes to attn_out to obtain <seqlen, dmodel> shape
               attn_out[i*nhead*dhead + n*dhead + j] = accum;
           }
       }
   }
}

void stage2_gt(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, int8_t* dense_weight_t, int32_t* dense_bias, float M_attention_probs, float M_attention_out, float M_dense_out, float M_residual, int16_t* norm_weight, int16_t* norm_bias, float M_stage2) {

    auto attn_score = new int32_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_probs = new int32_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_probs_int8 = new int8_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_out = new int32_t[CFG::seqlen*CFG::dmodel];
    auto attn_out_int8 = new int8_t[CFG::seqlen*CFG::dmodel];
    auto dense_out = new int32_t[CFG::dmodel*CFG::dmodel];
    auto dense_out_int8 = new int8_t[CFG::dmodel*CFG::dmodel];
    auto residual = new int16_t[CFG::seqlen*CFG::dmodel];
    auto ln_out = new int16_t[CFG::seqlen*CFG::dmodel];

    attention_scores(query_in, key_in, attn_score, CFG::seqlen, CFG::nhead, CFG::dhead);
    scale(attn_score);
    softmax(attn_score, attn_probs);
    requantize2(attn_probs, attn_probs_int8, 1, CFG::nhead*CFG::seqlen*CFG::seqlen, M_attention_probs);
    attention_values(attn_probs_int8, value_in, attn_out, CFG::seqlen, CFG::nhead, CFG::dhead);
    requantize2(attn_out, attn_out_int8, 1, CFG::seqlen*CFG::dmodel, M_attention_out);
    linear_sw2(attn_out_int8, dense_weight_t, dense_bias, dense_out, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    requantize2(dense_out, dense_out_int8, CFG::seqlen, CFG::dmodel, M_dense_out);
    add_skip2(dense_out_int8, skip_in, CFG::seqlen*CFG::dmodel);
    requantize2(dense_out_int8, residual, CFG::seqlen, CFG::dmodel, M_residual);
    layernorm_sw2(residual, ln_out, norm_weight, norm_bias, M_residual);
    requantize2(ln_out, stage2_out, CFG::seqlen, CFG::dmodel, M_stage2);

    delete[] attn_score;
    delete[] attn_probs;
    delete[] attn_probs_int8;
    delete[] attn_out;
    delete[] attn_out_int8;
    delete[] residual;
    delete[] ln_out;

}


void attention_scores_fused(int8_t* query, int8_t* key, int8_t* out, const int seqlen, const int nhead, const int dhead, float M_attention_probs) {

    int32_t divisor = std::sqrt(CFG::dmodel);
    int32_t rowbuff[CFG::seqlen];

    for (int n = 0; n < nhead; n++) {
        // compute matmul NHEAD times
        for (int i = 0; i < seqlen; i++) {
            for (int j = 0; j < seqlen; j++) {
                int32_t accum = 0;
                for (int k = 0; k < dhead; k++) {
                        // accum += query[n,i,k] * key[n, k, j]
                        accum += query[i*nhead*dhead +n*dhead + k] * key[j*nhead*dhead + n*dhead + k];
                }
                // out[n,i,j] = accum
                rowbuff[j] = accum / divisor;
            }
            softmax(rowbuff);
            for (int j = 0; j < seqlen; j++) {
                out[n*seqlen*seqlen + i*seqlen + j] = int8_t(rowbuff[j] * M_attention_probs);
            }
        }
    }
}

void attention_values_fused(int8_t* probs, int8_t* value, int8_t* attn_out, const int seqlen, const int nhead, const int dhead, float M_attention_out) {

    /**
     * probs: <nhead, seqlen, seqlen>
     * value: <seqlen, dmodel> -> <nhead, seqlen, dhead> (same reshape/transpose as query)
     * 
     * attn_out: <nhead, seqlen, dhead> -> <seqlen, dmodel> (how do you index to do this in one shot)
     * attn_out[i1][i2][i3] = attn_out[i1*seqlen*dhead + i2*dhead + i3]
     * att_out_transpose is <seqlen, nhead, dhead>
     * att_out_transpose[i1][i2][i3] = attn_out[i2][i1][i3] = attn_out[i2*nhead*dhead + i1*dhead + i3]
     * 
     * value_transpose[i1][i2][i3] = value[i2*nhead*dhead +i1*dhead + i3]
     * 
    */

   for (int n = 0; n < nhead; n++) {
       for (int i = 0; i < seqlen; i++) {
           for (int j = 0; j < dhead; j++) {
               int32_t accum = 0;
               for (int k = 0; k < seqlen; k++) {
                   // attn_out[n][i][j] += probs[n][i][k] * value[n][k][j]
                    accum += probs[n*seqlen*seqlen + i*seqlen + k] * value[k*nhead*dhead +n*dhead + j];
               }
               // writes to attn_out to obtain <seqlen, dmodel> shape
               attn_out[i*nhead*dhead + n*dhead + j] = int8_t(accum * M_attention_out);
           }
       }
   }
}

void linear_fused2(int8_t* A, int8_t* B, int32_t* bias, int16_t* out, const int N, const int M, const int K, int8_t* skip_conn, float M_dense, float M_residual) {
    
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

void layernorm_fused2(int16_t *act, int8_t *out, int16_t *norm_weight, int16_t *norm_bias, float scaling_factor, float M_stage)
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


void stage2(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, int8_t* dense_weight_t, int32_t* dense_bias, float M_attention_probs, float M_attention_out, float M_dense_out, float M_residual, int16_t* norm_weight, int16_t* norm_bias, float M_stage2) {
    int8_t att_scores_buff[CFG::nhead*CFG::seqlen*CFG::seqlen];
    int8_t att_out_buff[CFG::seqlen*CFG::dmodel];
    int16_t lin_buff[CFG::seqlen*CFG::dmodel];
    // attention, scale, softmax, and requantize
    attention_scores_fused(query_in, key_in, att_scores_buff, CFG::seqlen, CFG::nhead, CFG::dhead, M_attention_probs);
    // values, requantize
    attention_values_fused(att_scores_buff, value_in, att_out_buff, CFG::seqlen, CFG::nhead, CFG::dhead, M_attention_out);
    // linear, requantize, residual, requantize
    linear_fused2(att_out_buff, dense_weight_t, dense_bias, lin_buff, CFG::seqlen, CFG::dmodel, CFG::dmodel, skip_in, M_dense_out, M_residual);
    // layernorm, requantize
    layernorm_fused2(lin_buff, stage2_out, norm_weight, norm_bias, M_residual, M_stage2);

}
