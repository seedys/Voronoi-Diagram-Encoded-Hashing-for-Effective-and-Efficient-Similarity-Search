#ifndef GEMC_H
#define GEMC_H

#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <cstring>
#include <omp.h>
#include <tools/ResHandler.h>

#ifdef USE_AVX2
#include "tools/SimComputer_avx2.h"
#else
#include "tools/SimComputer.h"
#endif


void gemc_simd_parallel_1x4(const uint8_t *__restrict p1,
                                const uint8_t *__restrict p2,
                                size_t M, size_t N, size_t K,
                                size_t abs_idx1, size_t abs_idx2,
                                IFlib::BatchMinHeapHandler &res_handler);

void gemc_simd_8bits(const uint8_t *__restrict p1,
                                const uint8_t *__restrict p2,
                                size_t M, size_t N, size_t K,
                                size_t abs_idx1, size_t abs_idx2,
                                IFlib::BatchMinHeapHandler &res_handler);


void gemc_simd_bits(const uint8_t *__restrict p1,
                            const uint8_t *__restrict p2,
                            size_t M, size_t N, size_t K,
                            size_t abs_idx1, size_t abs_idx2,
                            IFlib::BatchMinHeapHandler &res_handler);

void gemc_simd_2bits(const uint8_t *__restrict p1,
                        const uint8_t *__restrict p2,
                        size_t M, size_t N, size_t K,
                        size_t abs_idx1, size_t abs_idx2,
                        IFlib::BatchMinHeapHandler &res_handler);

void gemc_simd_4bits(const uint8_t *__restrict p1,
                         const uint8_t *__restrict p2,
                         size_t M, size_t N, size_t K,
                         size_t abs_idx1, size_t abs_idx2,
                         IFlib::BatchMinHeapHandler &res_handler);


#endif