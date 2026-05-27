#include "IForest_index/gemc.h"

#define P1(i) (p1 + (i)*K)
#define P2(j) (p2 + (j)*K)
#define RES(i, j) res[ (i) * N + (j) ]
#define m256_SIMD_WIDTH 32 
#define SIMD_WIDTH 64
#define SIMD_WIDTH_x4 256

// Matrix comparison operation using SIMD, parallelization, and 1x4 configuration (direct __m512i* pointer usage, vectorized popcnt) (input data must be 64-byte aligned, K divisible by 256)
void gemc_simd_parallel_1x4(const uint8_t *__restrict p1,
                            const uint8_t *__restrict p2,
                            size_t M, size_t N, size_t K,
                            size_t abs_idx1, size_t abs_idx2,
                            IFlib::BatchMinHeapHandler &res_handler)
{

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i)
    {
        IFlib::BatchMinHeapHandler::SingleResultHandler res_single_handler(res_handler);
        res_single_handler.begin(abs_idx1 + i);
        const uint8_t *row_p1 = P1(i);

        size_t j = 0;
        for (; j + 4 <= N; j += 4)
        {
            // Base row pointer
            const __m512i *a_base = (const __m512i *)(row_p1);
            // 4-way query vector pointers
            const __m512i *b0 = (const __m512i *)(P2(j));
            const __m512i *b1 = (const __m512i *)(P2(j + 1));
            const __m512i *b2 = (const __m512i *)(P2(j + 2));
            const __m512i *b3 = (const __m512i *)(P2(j + 3));

            __m512i w, v0, v1, v2, v3, masks, counts;
            __m512i sumv = _mm512_setzero_si512();

            __mmask64 m0, m1, m2, m3;

            for (size_t k = 0; k < K; k += SIMD_WIDTH_x4)
            {
                // Prefetch next cache line
                _mm_prefetch((const char *)(a_base + 4), _MM_HINT_T0);
                _mm_prefetch((const char *)(b0 + 4), _MM_HINT_T0);
                _mm_prefetch((const char *)(b1 + 4), _MM_HINT_T0);
                _mm_prefetch((const char *)(b2 + 4), _MM_HINT_T0);
                _mm_prefetch((const char *)(b3 + 4), _MM_HINT_T0);

                // Load 64 bytes of data into 512-bit register
                w = _mm512_load_si512(a_base);

                v0 = _mm512_load_si512(b0);
                v1 = _mm512_load_si512(b1);
                v2 = _mm512_load_si512(b2);
                v3 = _mm512_load_si512(b3);

                // Compare 64 uint8_t and generate mask
                m0 = _mm512_cmpeq_epu8_mask(w, v0);
                m1 = _mm512_cmpeq_epu8_mask(w, v1);
                m2 = _mm512_cmpeq_epu8_mask(w, v2);
                m3 = _mm512_cmpeq_epu8_mask(w, v3);

                // Count number of 1s in 64-bit mask
                masks = _mm512_set_epi64(
                    0, 0, 0, 0,
                    (long long)m3,
                    (long long)m2,
                    (long long)m1,
                    (long long)m0);

                counts = _mm512_popcnt_epi64(masks);
                sumv = _mm512_add_epi64(sumv, counts);

                // Load 64 bytes of data into 512-bit register
                w = _mm512_load_si512(a_base + 1);

                v0 = _mm512_load_si512(b0 + 1);
                v1 = _mm512_load_si512(b1 + 1);
                v2 = _mm512_load_si512(b2 + 1);
                v3 = _mm512_load_si512(b3 + 1);

                // Compare 64 uint8_t and generate mask
                m0 = _mm512_cmpeq_epu8_mask(w, v0);
                m1 = _mm512_cmpeq_epu8_mask(w, v1);
                m2 = _mm512_cmpeq_epu8_mask(w, v2);
                m3 = _mm512_cmpeq_epu8_mask(w, v3);

                // Count number of 1s in 64-bit mask
                masks = _mm512_set_epi64(
                    0, 0, 0, 0,
                    (long long)m3,
                    (long long)m2,
                    (long long)m1,
                    (long long)m0);

                counts = _mm512_popcnt_epi64(masks);
                sumv = _mm512_add_epi64(sumv, counts);

                // Load 64 bytes of data into 512-bit register
                w = _mm512_load_si512(a_base + 2);

                v0 = _mm512_load_si512(b0 + 2);
                v1 = _mm512_load_si512(b1 + 2);
                v2 = _mm512_load_si512(b2 + 2);
                v3 = _mm512_load_si512(b3 + 2);

                // Compare 64 uint8_t and generate mask
                m0 = _mm512_cmpeq_epu8_mask(w, v0);
                m1 = _mm512_cmpeq_epu8_mask(w, v1);
                m2 = _mm512_cmpeq_epu8_mask(w, v2);
                m3 = _mm512_cmpeq_epu8_mask(w, v3);

                // Count number of 1s in 64-bit mask
                masks = _mm512_set_epi64(
                    0, 0, 0, 0,
                    (long long)m3,
                    (long long)m2,
                    (long long)m1,
                    (long long)m0);

                counts = _mm512_popcnt_epi64(masks);
                sumv = _mm512_add_epi64(sumv, counts);

                // Load 64 bytes of data into 512-bit register
                w = _mm512_load_si512(a_base + 3);

                v0 = _mm512_load_si512(b0 + 3);
                v1 = _mm512_load_si512(b1 + 3);
                v2 = _mm512_load_si512(b2 + 3);
                v3 = _mm512_load_si512(b3 + 3);

                // Compare 64 uint8_t and generate mask
                m0 = _mm512_cmpeq_epu8_mask(w, v0);
                m1 = _mm512_cmpeq_epu8_mask(w, v1);
                m2 = _mm512_cmpeq_epu8_mask(w, v2);
                m3 = _mm512_cmpeq_epu8_mask(w, v3);

                // Count number of 1s in 64-bit mask
                masks = _mm512_set_epi64(
                    0, 0, 0, 0,
                    (long long)m3,
                    (long long)m2,
                    (long long)m1,
                    (long long)m0);

                counts = _mm512_popcnt_epi64(masks);
                sumv = _mm512_add_epi64(sumv, counts);

                // Update pointers
                a_base += 4;
                b0 += 4;
                b1 += 4;
                b2 += 4;
                b3 += 4;
            }

            uint64_t tmp[8] __attribute__((aligned(64)));
            _mm512_store_si512((__m512i *)tmp, sumv);

            // Extract first 4 lanes
            res_single_handler.add_result(tmp[0], abs_idx2 + j);
            res_single_handler.add_result(tmp[1], abs_idx2 + j + 1);
            res_single_handler.add_result(tmp[2], abs_idx2 + j + 2);
            res_single_handler.add_result(tmp[3], abs_idx2 + j + 3);
        }

        // Process remainder when N is not divisible by 4
        for (; j < N; ++j)
        {
            long long sum = 0;
            __m512i v1, v2;
            const __m512i *a = (const __m512i *)(P1(i));
            const __m512i *b = (const __m512i *)(P2(j));
            __mmask64 cmp_mask;

            for (size_t k = 0; k < K; k += SIMD_WIDTH_x4)
            {
                // Load 64 bytes of data into 512-bit register
                v1 = _mm512_load_si512(a);
                v2 = _mm512_load_si512(b);

                // Compare 64 uint8_t and generate mask
                cmp_mask = _mm512_cmpeq_epu8_mask(v1, v2);
                sum += _mm_popcnt_u64(cmp_mask); // Count number of 1s in 64-bit mask

                // Load 64 bytes of data into 512-bit register
                v1 = _mm512_load_si512(a + 1);
                v2 = _mm512_load_si512(b + 1);

                // Compare 64 uint8_t and generate mask
                cmp_mask = _mm512_cmpeq_epu8_mask(v1, v2);
                sum += _mm_popcnt_u64(cmp_mask); // Count number of 1s in 64-bit mask

                // Load 64 bytes of data into 512-bit register
                v1 = _mm512_load_si512(a + 2);
                v2 = _mm512_load_si512(b + 2);

                // Compare 64 uint8_t and generate mask
                cmp_mask = _mm512_cmpeq_epu8_mask(v1, v2);
                sum += _mm_popcnt_u64(cmp_mask); // Count number of 1s in 64-bit mask

                // Load 64 bytes of data into 512-bit register
                v1 = _mm512_load_si512(a + 3);
                v2 = _mm512_load_si512(b + 3);

                // Compare 64 uint8_t and generate mask
                cmp_mask = _mm512_cmpeq_epu8_mask(v1, v2);
                sum += _mm_popcnt_u64(cmp_mask); // Count number of 1s in 64-bit mask

                // Update pointers
                a += 4;
                b += 4;
            }
            res_single_handler.add_result(sum, abs_idx2 + j);
        }
    }
}

void gemc_simd_8bits(const uint8_t *__restrict p1,
                        const uint8_t *__restrict p2,
                        size_t M, size_t N, size_t K,
                        size_t abs_idx1, size_t abs_idx2,
                        IFlib::BatchMinHeapHandler &res_handler)
{

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i)
    {
        IFlib::BatchMinHeapHandler::SingleResultHandler res_single_handler(res_handler);
        res_single_handler.begin(abs_idx1 + i);
        const uint8_t *row_p1 = P1(i);
        // int row_tmp[N];

        for (size_t j = 0; j <= N; j += 1)
        {
            const uint8_t *data2 = P2(j);
            int sim = IFlib::uint8_cmp(row_p1, data2, K);
            res_single_handler.add_result(sim, abs_idx2 + j);
        }
        
        // for (size_t j = 0; j < N; ++j)
        // {
        //     RES(i, j) = row_tmp[j];
        // }
    }
}

// Bit-level matrix comparison for 1x4 configuration (K = bytes per sample, must be 32-byte aligned, remainder truncated)
void gemc_simd_bits(const uint8_t *__restrict p1,
                           const uint8_t *__restrict p2,
                           size_t M, size_t N, size_t K, 
                           size_t abs_idx1, size_t abs_idx2, 
                           IFlib::BatchMinHeapHandler& res_handler)
{

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i)
    {
        IFlib::BatchMinHeapHandler::SingleResultHandler res_single_handler(res_handler);
        res_single_handler.begin(abs_idx1 + i);
        const uint8_t *row_p1 = P1(i);

        for (size_t j = 0; j < N; j += 1)
        {
            const uint8_t *data2 = P2(j);
            int sim = IFlib::bit_cmp(row_p1, data2, K);
            res_single_handler.add_result(sim, abs_idx2 + j);
        }
    } 
}

void gemc_simd_2bits(const uint8_t *__restrict p1,
                            const uint8_t *__restrict p2,
                            size_t M, size_t N, size_t K,
                            size_t abs_idx1, size_t abs_idx2,
                            IFlib::BatchMinHeapHandler &res_handler)
{

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i)
    {
        IFlib::BatchMinHeapHandler::SingleResultHandler res_single_handler(res_handler);
        res_single_handler.begin(abs_idx1 + i);
        const uint8_t *row_p1 = P1(i);

        for (size_t j = 0; j < N; j += 1)
        {
            const uint8_t *data2 = P2(j);
            int sim = IFlib::bit2_cmp_v2(row_p1, data2, K);
            res_single_handler.add_result(sim, abs_idx2 + j);
        }
    }
}

void gemc_simd_4bits(const uint8_t *__restrict p1,
                         const uint8_t *__restrict p2,
                         size_t M, size_t N, size_t K,
                         size_t abs_idx1, size_t abs_idx2,
                         IFlib::BatchMinHeapHandler &res_handler)
{

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i)
    {
        IFlib::BatchMinHeapHandler::SingleResultHandler res_single_handler(res_handler);
        res_single_handler.begin(abs_idx1 + i);
        const uint8_t *row_p1 = P1(i);

        for (size_t j = 0; j < N; j += 1)
        {
            const uint8_t *data2 = P2(j);
            int sim = IFlib::bit4_cmp_v2(row_p1, data2, K);
            res_single_handler.add_result(sim, abs_idx2 + j);
        }
    }
}
