#include "stage3.hpp"
#include "config.hpp"
#include <iostream>

void printmat(int8_t* A, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << int(A[i*N+j]) << ' ';
        }
        std::cout << std::endl;
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

    int8_t* fc_in = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t* fc_in_T = new int8_t[CFG::dmodel * CFG::seqlen];
    int8_t* dense_weight_t = new int8_t[CFG::dmodel * CFG::ffdim];
    int32_t* dense_bias = new int32_t[CFG::ffdim];

    genmat(fc_in, CFG::seqlen, CFG::dmodel, 7);
    for (int i = 0; i < CFG::dmodel; ++i) {
        for (int j = 0; j < CFG::seqlen; ++j) {
            fc_in_T[i*CFG::seqlen+j] = fc_in[j*CFG::dmodel+i];
        }
    }

    genmat(dense_weight_t, CFG::dmodel, CFG::ffdim, 13);

    genmat(dense_bias, 1, CFG::ffdim, 71);

    float dense_acc_scale = 0.004;
    float M_stage3 = 0.3;

    int8_t* dense_out_gt = new int8_t[CFG::seqlen * CFG::ffdim];
    int8_t* dense_out_test_T = new int8_t[CFG::seqlen * CFG::ffdim];
    int8_t* dense_out_test = new int8_t[CFG::seqlen * CFG::ffdim];

    stage3_gt(fc_in, dense_weight_t, dense_bias, dense_out_gt, dense_acc_scale, M_stage3);
    stage3(fc_in_T, dense_weight_t, dense_bias, dense_out_test_T, dense_acc_scale, M_stage3);

    for (int i = 0; i < CFG::seqlen; ++i) {
        for (int j = 0; j < CFG::ffdim; ++j) {
            dense_out_test[i*CFG::ffdim+j] = dense_out_test_T[j*CFG::seqlen+i];
        }
    }

    // printmat(dense_out, CFG::seqlen, CFG::ffdim);
    // std::cout << "\n\n\n\n";
    // printmat(dense_out_gt, CFG::seqlen, CFG::ffdim);

    std::cout << "dense_out: " << (check(dense_out_test, dense_out_gt, CFG::seqlen, CFG::ffdim) ? "PASSED" : "FAILED") << std::endl;

    delete [] fc_in;
    delete [] dense_weight_t;
    delete [] dense_bias;
    delete [] dense_out_test;
    delete [] dense_out_test_T;
    delete [] dense_out_gt;

    return 0;
}
