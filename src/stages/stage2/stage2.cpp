#include <cstdint>
#include <iostream>
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

void stage2_gt(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, float scores_scale, float M_attention_probs, float M_attention_out, float M_dense_out, int16_t* norm_weight, int16_t* norm_bias, float M_stage2) {


}
