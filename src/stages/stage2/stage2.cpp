#include <cstdint>
#include <iostream>
#include <cmath>
#include "../pipeline.hpp"

/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/
void linear_sw(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {
    
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

void requantize(int32_t* in, int8_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = int8_t(in[i*cols+j] * M_scale);
        }
    }
}

void scale(int32_t* y)
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
            //    attn_out[n*seqlen*dhead + i*dhead + j] = accum;
               attn_out[i*nhead*dhead + n*dhead + j] = accum;
           }
       }
   }
}

void stage2_gt(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, float M_attention_probs, float M_attention_out, float M_dense_out, int16_t* norm_weight, int16_t* norm_bias, float M_stage2) {

    auto attn_score = new int32_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_probs = new int32_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_probs_int8 = new int8_t[CFG::nhead*CFG::seqlen*CFG::seqlen];

    attention_scores(query_in, key_in, attn_score, CFG::seqlen, CFG::nhead, CFG::dhead);
    scale(attn_score);
    softmax(attn_score, attn_probs);
    requantize(attn_probs, attn_probs_int8, 1, CFG::nhead*CFG::seqlen*CFG::seqlen, M_attention_probs);


}
