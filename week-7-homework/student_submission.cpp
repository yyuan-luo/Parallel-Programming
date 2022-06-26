#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

#include <cstring>
#include <iostream>
#include <random>

#define MATRIX_SIZE 2017
#define ALIGNED_MATRIX_SIZE 2024
#define NUM_ELEMENTS MATRIX_SIZE * MATRIX_SIZE
#define MEM_SIZE NUM_ELEMENTS * sizeof(float)

void print256_num(__m256 var) {
    float val[8];
    memcpy(val, &var, sizeof(val));
    printf("%f %f %f %f %f %f %f %f \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}

void dgemm(float alpha, const float *a, const float *b, float beta, float *c) {
    __m256 be = _mm256_set1_ps(beta);
    __m256 al = _mm256_set1_ps(alpha);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < ALIGNED_MATRIX_SIZE; j += 8) {
            // c[i][j: j + 8]
            __m256 c_tmp = _mm256_load_ps(c + i * ALIGNED_MATRIX_SIZE + j);
            c_tmp = _mm256_mul_ps(c_tmp, be);
            for (int k = 0; k < MATRIX_SIZE; k++) {
                // a[i][k]
                __m256 a_tmp = _mm256_broadcast_ss(a + i * MATRIX_SIZE + k);
                a_tmp = _mm256_mul_ps(a_tmp, al);

                // b[k][j : j + 8]
                __m256 b_tmp = _mm256_load_ps(b + k * ALIGNED_MATRIX_SIZE + j);

                // * and +=
                __m256 tmp = _mm256_mul_ps(a_tmp, b_tmp);
                c_tmp = _mm256_add_ps(tmp, c_tmp);
//                printf("k: %d c[%d][%d: %d + 8]:\n", k, i, j, j);
//                print256_num(a_tmp);
//                print256_num(b_tmp);
//                print256_num(c_tmp);
            }
            _mm256_store_ps(c + i * ALIGNED_MATRIX_SIZE + j, c_tmp);
        }
    }
}

void generateProblemFromInput(float &alpha, float *a, float *b, float &beta, float *c) {
    unsigned int seed = 0;
    std::cout << "READY" << std::endl;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if (seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-5, 5);

    /* initialisation */
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            *(a + i * MATRIX_SIZE + j) = distribution(random);
            *(b + i * ALIGNED_MATRIX_SIZE + j) = distribution(random);
            *(c + i * ALIGNED_MATRIX_SIZE + j) = distribution(random);
        }
    }

    // TODO: simd assign value
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = MATRIX_SIZE; j < ALIGNED_MATRIX_SIZE; ++j) {
            *(b + i * ALIGNED_MATRIX_SIZE + j) = 0.0f;
            *(c + i * ALIGNED_MATRIX_SIZE + j) = 0.0f;
        }
    }

    alpha = distribution(random);
    beta = distribution(random);
//    for (int i = 0; i < MATRIX_SIZE; i++) {
//        for (int j = 0; j < MATRIX_SIZE; j++) {
//            printf("c[%d]: %f  ", i * MATRIX_SIZE + j, *(c + i * MATRIX_SIZE + j));
////            *(c + i * ALIGNED_MATRIX_SIZE + j)
//        }
//        printf("\n");
//    }
//
//    for (int i = 0; i < MATRIX_SIZE; i++) {
//        for (int j = 0; j < MATRIX_SIZE; j++) {
//            printf("b[%d]: %f  ", i * MATRIX_SIZE + j, *(b + i * MATRIX_SIZE + j));
////            *(c + i * ALIGNED_MATRIX_SIZE + j)
//        }
//        printf("\n");
//    }
//
//    for (int i = 0; i < MATRIX_SIZE; i++) {
//        for (int j = 0; j < MATRIX_SIZE; j++) {
//            printf("a[%d]: %f  ", i * MATRIX_SIZE + j, *(a + i * MATRIX_SIZE + j));
////            *(c + i * ALIGNED_MATRIX_SIZE + j)
//        }
//        printf("\n");
//    }
}

void outputSolution(const float *c) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < 2017 * 2024; ++i) {
//        printf("c[%d]: %f ", i, c[i]);
//        if (i % 300 == 0)
//            printf("\n");
        sum += c[i] * ((i + 1) % 7);
    }
//    for (int i = 0; i < MATRIX_SIZE; ++i) {
//        for (int j = 0; j < MATRIX_SIZE; ++j) {
//            sum += c[i * MATRIX_SIZE + j] * ((i * MATRIX_SIZE + j) % 7);
//        }
//    }

    std::cout << "Sum of final matrix values: " << sum << std::endl;
    std::cout << "DONE" << std::endl;
}

int main(int, char **) {
    float alpha, beta;

    // mem allocations
    int mem_size = MATRIX_SIZE * ALIGNED_MATRIX_SIZE * sizeof(float);
    float *a = (float *) _mm_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 32);
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
