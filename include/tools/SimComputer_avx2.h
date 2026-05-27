#ifndef SIM_COMPUTER_AVX2_H
#define SIM_COMPUTER_AVX2_H

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

namespace IFlib
{
    template <typename Ret = int>
    inline Ret uint8_cmp(const uint8_t *__restrict p1,
                         const uint8_t *__restrict p2,
                         size_t K)
    {
        throw std::logic_error("uint8_cmp is not implemented yet");
    }

    // Count how many 1 bits are in a byte
    static inline __m256i avx2_popcnt_bytes(__m256i x)
    {
        const __m256i lut = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        __m256i lo = _mm256_and_si256(x, m4);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m4);
        __m256i pc_lo = _mm256_shuffle_epi8(lut, lo);
        __m256i pc_hi = _mm256_shuffle_epi8(lut, hi);
        return _mm256_add_epi8(pc_lo, pc_hi);
    }

    template <typename Ret = int>
    inline Ret bit_cmp(const uint8_t *__restrict p1,
                            const uint8_t *__restrict p2,
                            size_t K)
    {
        // please make sure K % 32 == 0
        const size_t n_feature = K * 8;

        const uint8_t *a8 = p1;
        const uint8_t *b8 = p2;

        size_t k = 0;
        __m256i acc = _mm256_setzero_si256();
        const __m256i zero256 = _mm256_setzero_si256();

        // 4x unroll: Processing 128 bytes each time
        for (; k + 32 * 4 <= K; k += 32 * 4)
        {
            __m256i a0 = _mm256_loadu_si256((const __m256i *)(a8 + 0));
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(b8 + 0));
            __m256i a1 = _mm256_loadu_si256((const __m256i *)(a8 + 32));
            __m256i b1 = _mm256_loadu_si256((const __m256i *)(b8 + 32));
            __m256i a2 = _mm256_loadu_si256((const __m256i *)(a8 + 64));
            __m256i b2 = _mm256_loadu_si256((const __m256i *)(b8 + 64));
            __m256i a3 = _mm256_loadu_si256((const __m256i *)(a8 + 96));
            __m256i b3 = _mm256_loadu_si256((const __m256i *)(b8 + 96));

            __m256i d0 = _mm256_xor_si256(a0, b0);
            __m256i d1 = _mm256_xor_si256(a1, b1);
            __m256i d2 = _mm256_xor_si256(a2, b2);
            __m256i d3 = _mm256_xor_si256(a3, b3);

            __m256i pc0 = avx2_popcnt_bytes(d0);
            __m256i pc1 = avx2_popcnt_bytes(d1);
            __m256i pc2 = avx2_popcnt_bytes(d2);
            __m256i pc3 = avx2_popcnt_bytes(d3);

            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc0, zero256));
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc1, zero256));
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc2, zero256));
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc3, zero256));

            a8 += 32 * 4;
            b8 += 32 * 4;
        }

        // Process the remaining 32 byte block
        for (; k + 32 <= K; k += 32)
        {
            __m256i a = _mm256_loadu_si256((const __m256i *)a8);
            __m256i b = _mm256_loadu_si256((const __m256i *)b8);
            __m256i d = _mm256_xor_si256(a, b);
            __m256i pc = avx2_popcnt_bytes(d);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc, zero256));
            a8 += 32;
            b8 += 32;
        }

        __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
        __m128i acc_lo = _mm256_castsi256_si128(acc);
        __m128i acc_sum128 = _mm_add_epi64(acc_lo, acc_hi);
        uint64_t diff_bits =
            (uint64_t)_mm_cvtsi128_si64(acc_sum128) +
            (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc_sum128, acc_sum128));

        return static_cast<Ret>(n_feature - diff_bits);
    }


    template <typename Ret = int>
    inline Ret bit2_cmp_v2(const uint8_t *__restrict p1,
                           const uint8_t *__restrict p2,
                           size_t K)
    {
        constexpr size_t SIMD_WIDTH = 32;
        size_t n_bits = K * 8;
        size_t k = 0;

        const __m256i *a = (const __m256i *)(p1);
        const __m256i *b = (const __m256i *)(p2);
        // 10101010 = 170
        __m256i or_mask = _mm256_set1_epi8(170);
        __m256i sumv = _mm256_setzero_si256();
        const __m256i zero256 = _mm256_setzero_si256();

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {

            __m256i a0 = _mm256_loadu_si256(a);
            __m256i b0 = _mm256_loadu_si256(b);
            //  XOR operation
            __m256i diff0 = _mm256_xor_si256(a0, b0);

            __m256i diff0_r1 = _mm256_srli_epi64(diff0, 1);

            __m256i mask = _mm256_or_si256(diff0, diff0_r1);

            __m256i final = _mm256_or_si256(mask, or_mask);

            __m256i popcnt_vec = avx2_popcnt_bytes(final);

            // 64 bit add as a whole
            sumv = _mm256_add_epi64(sumv, _mm256_sad_epu8(popcnt_vec, zero256));

            a += 1;
            b += 1;
        }

        __m128i acc_hi = _mm256_extracti128_si256(sumv, 1);
        __m128i acc_lo = _mm256_castsi256_si128(sumv);
        __m128i acc_sum128 = _mm_add_epi64(acc_lo, acc_hi);
        uint64_t diff_bits =
            (uint64_t)_mm_cvtsi128_si64(acc_sum128) +
            (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc_sum128, acc_sum128));
        return static_cast<Ret>(n_bits - diff_bits);
    }

    template <typename Ret = int>
    inline Ret bit4_cmp_v2(const uint8_t *__restrict p1,
                           const uint8_t *__restrict p2,
                           size_t K)
    {
        constexpr size_t SIMD_WIDTH = 32;
        size_t n_bits = K * 8;
        size_t k = 0;

        const __m256i *a = (const __m256i *)(p1);
        const __m256i *b = (const __m256i *)(p2);
        // 10101010 = 170
        __m256i or_mask = _mm256_set1_epi8(238);
        __m256i sumv = _mm256_setzero_si256();
        const __m256i zero256 = _mm256_setzero_si256();

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {

            __m256i a0 = _mm256_loadu_si256(a);
            __m256i b0 = _mm256_loadu_si256(b);
            //  XOR operation
            __m256i diff0 = _mm256_xor_si256(a0, b0);

            __m256i diff0_r1 = _mm256_srli_epi64(diff0, 1);
            __m256i diff0_r2 = _mm256_srli_epi64(diff0_r1, 1);
            __m256i diff0_r3 = _mm256_srli_epi64(diff0_r2, 1);

            __m256i mask1 = _mm256_or_si256(diff0, diff0_r1);
            __m256i mask2 = _mm256_or_si256(diff0_r2, diff0_r3);
            __m256i mask = _mm256_or_si256(mask1, mask2);

            __m256i final = _mm256_or_si256(mask, or_mask);

            __m256i popcnt_vec = avx2_popcnt_bytes(final);

            // 64 bits added as a whole
            sumv = _mm256_add_epi64(sumv, _mm256_sad_epu8(popcnt_vec, zero256));

            a += 1;
            b += 1;
        }

        __m128i acc_hi = _mm256_extracti128_si256(sumv, 1);
        __m128i acc_lo = _mm256_castsi256_si128(sumv);
        __m128i acc_sum128 = _mm_add_epi64(acc_lo, acc_hi);
        uint64_t diff_bits =
            (uint64_t)_mm_cvtsi128_si64(acc_sum128) +
            (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc_sum128, acc_sum128));
        return static_cast<Ret>(n_bits - diff_bits);
    }

    struct IForestSimComputer
    {
        size_t d;                 // dim
        const uint8_t *base_data; // base data
        const uint8_t *q;         // query data

        explicit IForestSimComputer(size_t d, const uint8_t *x) : d(d)
        {
            base_data = x;
        }

        void set_query(const uint8_t *x)
        {
            q = x;
        }

        virtual int symmetric_sim(size_t i, size_t j) = 0;

        virtual int operator()(size_t i) = 0;

        void sim_batch_4(
            const size_t idx0,
            const size_t idx1,
            const size_t idx2,
            const size_t idx3,
            int &sim0,
            int &sim1,
            int &sim2,
            int &sim3)
        {
            // compute first, assign next
            const int d0 = this->operator()(idx0);
            const int d1 = this->operator()(idx1);
            const int d2 = this->operator()(idx2);
            const int d3 = this->operator()(idx3);
            sim0 = d0;
            sim1 = d1;
            sim2 = d2;
            sim3 = d3;
        }
    };

    struct IForestSimComputerUint8 : public IForestSimComputer
    {
        explicit IForestSimComputerUint8(size_t d, const uint8_t *x) : IForestSimComputer(d, x) {}

        int symmetric_sim(size_t i, size_t j) override
        {
            const uint8_t *pi = base_data + i * d;
            const uint8_t *pj = base_data + j * d;

            return uint8_cmp(pi, pj, d);
        }

        int operator()(size_t i) override
        {
            const uint8_t *pi = base_data + i * d;
            return uint8_cmp(q, pi, d);
        }
    };

    struct IForestSimComputerBit : public IForestSimComputer
    {
        explicit IForestSimComputerBit(size_t d, const uint8_t *x) : IForestSimComputer(d, x) {}

        int symmetric_sim(size_t i, size_t j) override
        {
            const uint8_t *pi = base_data + i * d;
            const uint8_t *pj = base_data + j * d;

            return bit_cmp(pi, pj, d);
        }

        int operator()(size_t i) override
        {
            const uint8_t *pi = base_data + i * d;
            return bit_cmp(q, pi, d);
        }
    };

    struct IForestSimComputer2Bit : public IForestSimComputer
    {
        explicit IForestSimComputer2Bit(size_t d, const uint8_t *x) : IForestSimComputer(d, x) {}

        int symmetric_sim(size_t i, size_t j) override
        {
            const uint8_t *pi = base_data + i * d;
            const uint8_t *pj = base_data + j * d;

            return bit2_cmp_v2(pi, pj, d);
        }

        int operator()(size_t i) override
        {
            const uint8_t *pi = base_data + i * d;
            return bit2_cmp_v2(q, pi, d);
        }
    };

    struct IForestSimComputer4Bit : public IForestSimComputer
    {
        explicit IForestSimComputer4Bit(size_t d, const uint8_t *x) : IForestSimComputer(d, x) {}

        int symmetric_sim(size_t i, size_t j) override
        {
            const uint8_t *pi = base_data + i * d;
            const uint8_t *pj = base_data + j * d;

            return bit4_cmp_v2(pi, pj, d);
        }

        int operator()(size_t i) override
        {
            const uint8_t *pi = base_data + i * d;
            return bit4_cmp_v2(q, pi, d);
        }
    };

} // namespace IFlib

#endif // !SIM_COMPUTER_AVX2_H