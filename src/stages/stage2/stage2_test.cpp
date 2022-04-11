#include "stage2.hpp"
#include "../pipeline.hpp"
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

    int8_t query_in[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23};
    int8_t key_in[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23};
    auto att_out_test = new int32_t [3*2*2];

    int32_t att_out_gt[] = {14,   86,   86,  734,  126,  390,  390, 1230,  366,  822,  822, 1854};

    genmat(query_in, 1, 24, 100);
    genmat(key_in, 1, 24, 100);

    attention_scores(query_in, key_in, att_out_test, 2, 3, 4);

    // delete[] query_in;
    // delete[] key_in;
    delete[] att_out_test;

    std::cout << "att_out: " << (check(att_out_gt, att_out_test, 1, (3*2*2)) ? "PASSED" : "FAILED") << std::endl;
    // std::cout << "key_out:   " << (check(key_out_gt, key_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    // std::cout << "value_out: " << (check(value_out_gt, value_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    return 0;
}