#include "dgemm.h"
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

void dgemm(float alpha, float *a, const float *b, float beta, float *c) {
//    for (int i = 0; i < MATRIX_SIZE; i++) {
//        for (int j = 0; j < MATRIX_SIZE; j++) {
//            c[i * MATRIX_SIZE + j] *= beta;
//            for (int k = 0; k < MATRIX_SIZE; k++) {
//                c[i * MATRIX_SIZE + j] += alpha * a[i * MATRIX_SIZE + k] * b[j * MATRIX_SIZE + k];
//            }
//        }
//    }
    __m256 be = _mm256_set1_ps(beta);
    __m256 al = _mm256_set1_ps(alpha);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE - 8; i += 8) {
        __m256 c_tmp = _mm256_load_ps(c + i);
        __m256 a_tmp = _mm256_load_ps(a + i);
        c_tmp = _mm256_mul_ps(c_tmp, be);
        a_tmp = _mm256_mul_ps(a_tmp, al);
        _mm256_store_ps(c + i, c_tmp);
        _mm256_store_ps(a + i, a_tmp);
    }
    // deal with the reminder
    c[4068288] *= beta;
    a[4068288] *= alpha;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            for (int k = 0; k < MATRIX_SIZE; k++) {
                c[i * MATRIX_SIZE + j] += a[i * MATRIX_SIZE + k] * b[j * MATRIX_SIZE + k];
            }
        }
    }
}

int main(int, char **) {
    float alpha, beta;

    // mem allocations
    int mem_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    float *a = (float *) _mm_malloc(mem_size, 32);
    float *b = (float *) _mm_malloc(mem_size, 32);
    float *c = (float *) _mm_malloc(mem_size, 32);

    // check if allocated
    if (nullptr == a || nullptr == b || nullptr == c) {
        printf("Memory allocation failed\n");
        if (nullptr != a) free(a);
        if (nullptr != b) free(b);
        if (nullptr != c) free(c);
        return 0;
    }

    generateProblemFromInput(alpha, a, b, beta, c);

    std::cerr << "Launching dgemm step." << std::endl;
    // matrix-multiplication
    dgemm(alpha, a, b, beta, c);

    outputSolution(c);

    free(a);
    free(b);
    free(c);
    return 0;
}
