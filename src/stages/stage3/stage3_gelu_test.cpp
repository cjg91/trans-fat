#include "stage3.hpp"
#include <iostream>

template<typename T>
void printmat(T* A, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << int32_t(A[i*N+j]) << ' ';
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
    int32_t gelu_in[] = {-3, -1, 1, 3, 5};
    float scale = 0.01;
    int32_t gelu_gt[] = {-0,   200,  -700, -2700, -4500};

    int32_t gelu_test[5];

    gelu_sw(gelu_in, gelu_test, 1, 5, scale);

    std::cout << "gelu: " << (check(gelu_gt, gelu_test, 1, 5) ? "PASSED" : "FAILED") << std::endl;

    printmat(gelu_test, 1, 5);
}