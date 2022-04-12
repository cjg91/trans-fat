#include "stage2.hpp"
#include "../../config.hpp"
#include <iostream>

template<typename T>
void printmat(T* A, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << int(A[i*N+j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_query_transpose(int8_t* Q, const int seqlen, const int nhead, const int dhead) {

    for (int n = 0; n < nhead; n++) {
        for (int i = 0; i < seqlen; i++) {
            for (int j = 0; j < dhead; j++) {
                // query[i2*nhead*dhead +i1*dhead + i3]
                std::cout << int(Q[i*nhead*dhead +n*dhead + j]) << ' ';
            }
        }
    }
    std::cout << std::endl;
}

void print_key_transpose(int8_t* K, const int seqlen, const int nhead, const int dhead) {
    // <nhead, dhead, seqlen

    for (int n = 0; n < nhead; n++) {
        for (int i = 0; i < dhead; i++) {
            for (int j = 0; j < seqlen; j++) {
                // key[i2*nhead*dhead + i3*dhead + i1] (supposedly)
                std::cout << int(K[j*nhead*dhead + n*dhead + i]) << ' ';
            }
        }
    }
    std::cout << std::endl;
}

template<typename T>
void genmat(T* A, const int M, const int N, const int mod) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N+j] = (i*N+j) % mod;
        }
    }
}

template<typename T>
const bool check(T* A, T* B, const int M, const int N)
{
    for (int i = 0; i < M*N; i++) {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

int main() {

    const int nhead = 2;
    const int seqlen = 3;
    const int dhead = 4; 

    /**
     * Attention Scores Test
    */
    auto query_in = new int8_t[nhead*seqlen*dhead];
    auto key_in = new int8_t[nhead*seqlen*dhead];

    genmat(query_in, 1, nhead*seqlen*dhead, nhead*seqlen*dhead+1);
    genmat(key_in, 1, nhead*seqlen*dhead, nhead*seqlen*dhead+1);

    auto att_score_test = new int32_t [seqlen*nhead*nhead];
    int32_t att_score_gt[] = {14,   86,   86,  734,  126,  390,  390, 1230,  366,  822,  822, 1854};

    attention_scores(query_in, key_in, att_score_test, 2, 3, 4);
    
    std::cout << "att_out: " << (check(att_score_gt, att_score_test, 1, (3*2*2)) ? "PASSED" : "FAILED") << std::endl;

    delete[] query_in;
    delete[] key_in;
    delete[] att_score_test;

    /**
     * Attention Values Test (Probs * Values)
    */

    auto value_in = new int8_t[nhead*seqlen*dhead];
    auto probs_in = new int8_t[nhead*seqlen*seqlen];

    genmat(value_in, 1, nhead*seqlen*dhead, nhead*seqlen*dhead+1);
    genmat(probs_in, 1, nhead*seqlen*seqlen, nhead*seqlen*seqlen+1);

    auto att_out_test = new int32_t [nhead*seqlen*dhead];
    int32_t att_out_gt[] = { 40,  43,  46,  49, 376, 406, 436, 466, 112, 124, 136, 148, 484, 523,
        562, 601, 184, 205, 226, 247, 592, 640, 688, 736};

    attention_values(probs_in, value_in, att_out_test, seqlen, nhead, dhead);

    std::cout << "att_out: " << (check(att_out_gt, att_out_test, 1, seqlen*nhead*dhead) ? "PASSED" : "FAILED") << std::endl;


    delete[] value_in;
    delete[] probs_in;
    delete[] att_out_test;



    /**
     * 
     * Test Fused Correctness
     * 
    */

    query_in = new int8_t[CFG::seqlen*CFG::dmodel];
    key_in = new int8_t[CFG::seqlen*CFG::dmodel];
    value_in = new int8_t[CFG::seqlen*CFG::dmodel];
    auto skip_in = new int8_t[CFG::seqlen*CFG::dmodel];
    auto dense_weight_t = new int8_t[CFG::dmodel*CFG::dmodel];
    auto dense_bias = new int32_t[CFG::dmodel];
    auto norm_weight = new int16_t[CFG::dmodel];
    auto norm_bias = new int16_t[CFG::dmodel];
    auto stage2_out = new int8_t[CFG::seqlen*CFG::dmodel];

    genmat(query_in, CFG::seqlen, CFG::dmodel, 7);
    genmat(key_in, CFG::seqlen, CFG::dmodel, 9);
    genmat(value_in, CFG::seqlen, CFG::dmodel, 11);
    genmat(skip_in, CFG::seqlen, CFG::dmodel, 13);
    genmat(dense_weight_t, CFG::dmodel, CFG::dmodel, 15);
    genmat(dense_bias, 1, CFG::dmodel, 63);
    genmat(norm_weight, 1, CFG::dmodel, 17);
    genmat(norm_bias, 1, CFG::dmodel, 23);

    float M_attention_probs = 0.1;
    float M_attention_out = 0.1;
    float M_dense_out = 0.1;
    float M_residual = 1;
    float M_stage2 = 1;


    stage2_gt(query_in, key_in, value_in, skip_in, stage2_out, dense_weight_t, dense_bias, M_attention_probs, M_attention_out, M_dense_out, M_residual, norm_weight, norm_bias, M_stage2);

    printmat(stage2_out, CFG::seqlen, CFG::dmodel);


    return 0;
}