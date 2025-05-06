#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <arm_neon.h>

#define TEST_LEN        10240000

#define REPEAT_TIMES    10

void neon_add_sub_v1_optimized(float* colsum, float* x_add, float* x_sub, float* dst, int n)
{
    n = n >> 2;

    asm volatile(
        "0:                                     \n"

        "vld1.32 {d0-d1}, [%0]                  \n"
        "vld1.32 {d2-d3}, [%1]!                 \n"
        "vld1.32 {d4-d5}, [%2]!                 \n"

        "vadd.f32 q0, q0, q1                    \n"
        "vsub.f32 q0, q0, q2                    \n"

        "vst1.32 {d0-d1}, [%0]!                 \n" // store result to colsum
        "vst1.32 {d0-d1}, [%3]!                 \n" // store result to dst

        "subs     %4, #1                        \n"
        "bne      0b                            \n"
        : "=r"(colsum), "=r"(x_add), "=r"(x_sub), "=r"(dst), "=r"(n)
        : "0"(colsum), "1"(x_add), "2"(x_sub), "3"(dst), "4"(n)
        : "cc", "memory", "q0", "q1", "q2");
}


void neon_add_sub_v2(float* colsum, float* x_add, float* x_sub, float* dst, int n)
{
    for (int i = 0; i < n; i++) {
        colsum[i] += x_add[i];
        colsum[i] -= x_sub[i];
        dst[i] = colsum[i];
    }
}


int main(void)
{
#if 0
    float colsum[TEST_LEN] __attribute__((aligned(64))) = {0};
    float x_add[TEST_LEN]  __attribute__((aligned(64))) = {0};
    float x_sub[TEST_LEN]  __attribute__((aligned(64))) = {0};
    float dst[TEST_LEN]    __attribute__((aligned(64))) = {0};
#else
    float *colsum = malloc(TEST_LEN * sizeof(float));
    float *x_add = malloc(TEST_LEN * sizeof(float));
    float *x_sub = malloc(TEST_LEN * sizeof(float));
    float *dst = malloc(TEST_LEN * sizeof(float));
#endif

    for (int i = 0; i < TEST_LEN; i++) {
        colsum[i] = 1.0f;
        x_add[i] = 0.5f;
        x_sub[i] = 0.25f;
    }

    int n = TEST_LEN;

    uint32_t start1 = clock();
    for (int i = 0; i < REPEAT_TIMES; i++) {
        neon_add_sub_v1_optimized(colsum, x_add, x_sub, dst, n);
    }
    uint32_t end1 = clock();

    for (int i = 0; i < TEST_LEN; i++) {
        if (colsum[i] != REPEAT_TIMES * 0.25f + 1.0f) {
            printf("colsum[%d] = %f\n", i, colsum[i]);
        }
        colsum[i] = 1.0f;
    }

    uint32_t start2 = clock();
    for (int i = 0; i < REPEAT_TIMES; i++) {
        neon_add_sub_v2(colsum, x_add, x_sub, dst, n);
    }
    uint32_t end2 = clock();

    for (int i = 0; i < TEST_LEN; i++) {
        if (colsum[i] != REPEAT_TIMES * 0.25f + 1.0f) {
            printf("colsum[%d] = %f\n", i, colsum[i]);
        }
        colsum[i] = 1.0f;
    }

    printf("v1 cost : %f ms\n", (double)(end1 - start1) / CLOCKS_PER_SEC * 1000);
    printf("v2 cost : %f ms\n", (double)(end2 - start2) / CLOCKS_PER_SEC * 1000);

    return 0;
}
