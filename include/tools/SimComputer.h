#ifndef SIM_COMPUTER_H
#define SIM_COMPUTER_H

/*
This .h file contains functions for calculating the similarity between two IKE data points,
where the bit"n"_cmp functions are used to compute similarity between IKE data points where
each element is compressed into n bits.

Note: The parameter K represents the code length (in bytes) of an IKE data point. 
Since the functions in this file are implemented using AVX512 instructions, K must be divisible by 64.
*/

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace IFlib{
    template <typename Ret = int>
    inline Ret scalar_cmp(const uint8_t *__restrict p1,
                         const uint8_t *__restrict p2,
                         size_t K)
    {
        int sum = 0;
        for (size_t i = 0; i < K; i++){
            if(p1[i] == p2[i])
                sum += 1;
        }

        return static_cast<Ret>(sum);
    }

    template <typename Ret = int>
    inline Ret uint8_cmp(const uint8_t *__restrict p1,
                         const uint8_t *__restrict p2,
                         size_t K)
    {
        constexpr size_t SIMD_WIDTH_x4 = 256;
    
        const __m512i *a_base = (const __m512i *)(p1);
        const __m512i *b_base = (const __m512i *)(p2);

        __m512i w, v;
        __mmask64 m;

        long long sum = 0;

        for (size_t k = 0; k < K; k += SIMD_WIDTH_x4)
        {
            // Load 64 bytes of data into a 512 bit register
            w = _mm512_loadu_si512(a_base);
            v = _mm512_loadu_si512(b_base);

            // Compare 64 uint8_t and generate a mask
            m = _mm512_cmpeq_epu8_mask(w, v);
            sum += _mm_popcnt_u64(m);

            w = _mm512_loadu_si512(a_base + 1);
            v = _mm512_loadu_si512(b_base + 1);

            m = _mm512_cmpeq_epu8_mask(w, v);
            sum += _mm_popcnt_u64(m);

            w = _mm512_loadu_si512(a_base + 2);
            v = _mm512_loadu_si512(b_base + 2);

            m = _mm512_cmpeq_epu8_mask(w, v);
            sum += _mm_popcnt_u64(m);

            w = _mm512_loadu_si512(a_base + 3);
            v = _mm512_loadu_si512(b_base + 3);

            m = _mm512_cmpeq_epu8_mask(w, v);
            sum += _mm_popcnt_u64(m);

            // update pointers
            a_base += 4;
            b_base += 4;
        }

        return static_cast<Ret>(sum);
    }

    inline void uint8_cmp_1x4(const uint8_t *__restrict q,
                             const uint8_t *__restrict p0,
                             const uint8_t *__restrict p1,
                             const uint8_t *__restrict p2,
                             const uint8_t *__restrict p3,
                             size_t K,
                             int &sim0,
                             int &sim1,
                             int &sim2,
                             int &sim3)
    {
        constexpr size_t SIMD_WIDTH_x4 = 256;
        alignas(64) long long tmp[8];

        // base data pointer
        const __m512i *a_base = (const __m512i *)(q);
        // 4 query pointers
        const __m512i *b0 = (const __m512i *)(p0);
        const __m512i *b1 = (const __m512i *)(p1);
        const __m512i *b2 = (const __m512i *)(p2);
        const __m512i *b3 = (const __m512i *)(p3);

        __m512i w, v0, v1, v2, v3, masks, counts;
        __m512i sumv = _mm512_setzero_si512();

        __mmask64 m0, m1, m2, m3;

        for (size_t k = 0; k < K; k += SIMD_WIDTH_x4)
        {
            // Load 64 bytes of data into a 512 bit register
            w = _mm512_load_si512(a_base);

            v0 = _mm512_load_si512(b0);
            v1 = _mm512_load_si512(b1);
            v2 = _mm512_load_si512(b2);
            v3 = _mm512_load_si512(b3);

            // Compare 64 uint8_t and generate a mask
            m0 = _mm512_cmpeq_epu8_mask(w, v0);
            m1 = _mm512_cmpeq_epu8_mask(w, v1);
            m2 = _mm512_cmpeq_epu8_mask(w, v2);
            m3 = _mm512_cmpeq_epu8_mask(w, v3);

            // Count the number of 1 bit in a 64 bit mask
            masks = _mm512_set_epi64(
                0, 0, 0, 0,
                (long long)m3,
                (long long)m2,
                (long long)m1,
                (long long)m0);

            counts = _mm512_popcnt_epi64(masks);
            sumv = _mm512_add_epi64(sumv, counts);


            w = _mm512_load_si512(a_base + 1);

            v0 = _mm512_load_si512(b0 + 1);
            v1 = _mm512_load_si512(b1 + 1);
            v2 = _mm512_load_si512(b2 + 1);
            v3 = _mm512_load_si512(b3 + 1);

            m0 = _mm512_cmpeq_epu8_mask(w, v0);
            m1 = _mm512_cmpeq_epu8_mask(w, v1);
            m2 = _mm512_cmpeq_epu8_mask(w, v2);
            m3 = _mm512_cmpeq_epu8_mask(w, v3);

            masks = _mm512_set_epi64(
                0, 0, 0, 0,
                (long long)m3,
                (long long)m2,
                (long long)m1,
                (long long)m0);

            counts = _mm512_popcnt_epi64(masks);
            sumv = _mm512_add_epi64(sumv, counts);


            w = _mm512_load_si512(a_base + 2);

            v0 = _mm512_load_si512(b0 + 2);
            v1 = _mm512_load_si512(b1 + 2);
            v2 = _mm512_load_si512(b2 + 2);
            v3 = _mm512_load_si512(b3 + 2);

            m0 = _mm512_cmpeq_epu8_mask(w, v0);
            m1 = _mm512_cmpeq_epu8_mask(w, v1);
            m2 = _mm512_cmpeq_epu8_mask(w, v2);
            m3 = _mm512_cmpeq_epu8_mask(w, v3);

            masks = _mm512_set_epi64(
                0, 0, 0, 0,
                (long long)m3,
                (long long)m2,
                (long long)m1,
                (long long)m0);

            counts = _mm512_popcnt_epi64(masks);
            sumv = _mm512_add_epi64(sumv, counts);


            w = _mm512_load_si512(a_base + 3);

            v0 = _mm512_load_si512(b0 + 3);
            v1 = _mm512_load_si512(b1 + 3);
            v2 = _mm512_load_si512(b2 + 3);
            v3 = _mm512_load_si512(b3 + 3);

            m0 = _mm512_cmpeq_epu8_mask(w, v0);
            m1 = _mm512_cmpeq_epu8_mask(w, v1);
            m2 = _mm512_cmpeq_epu8_mask(w, v2);
            m3 = _mm512_cmpeq_epu8_mask(w, v3);

            masks = _mm512_set_epi64(
                0, 0, 0, 0,
                (long long)m3,
                (long long)m2,
                (long long)m1,
                (long long)m0);

            counts = _mm512_popcnt_epi64(masks);
            sumv = _mm512_add_epi64(sumv, counts);

            // update pointers
            a_base += 4;
            b0 += 4;
            b1 += 4;
            b2 += 4;
            b3 += 4;
        }

        _mm512_store_si512((__m512i *)tmp, sumv);

        sim0 = static_cast<int>(tmp[0]);
        sim1 = static_cast<int>(tmp[1]);
        sim2 = static_cast<int>(tmp[2]);
        sim3 = static_cast<int>(tmp[3]);
    }

    template <typename Ret = int>
    inline Ret bit_cmp(const uint8_t *__restrict p1,
                       const uint8_t *__restrict p2,
                       size_t K)
    {
        constexpr size_t SIMD_WIDTH_x4 = 256;
        constexpr size_t SIMD_WIDTH = 64;
        constexpr size_t m256_SIMD_WIDTH = 32;
        size_t n_feature = K * 8;

        const __m512i *a = (const __m512i *)(p1);
        const __m512i *b = (const __m512i *)(p2);

        size_t k = 0;
        
        __m512i sumv0 = _mm512_setzero_si512();
        __m512i sumv1 = _mm512_setzero_si512();
        __m512i sumv2 = _mm512_setzero_si512();
        __m512i sumv3 = _mm512_setzero_si512();

        // unroll
        for (; k + SIMD_WIDTH_x4 <= K; k += SIMD_WIDTH_x4)
        {
            // 1. First, load all the data
            __m512i a0 = _mm512_loadu_si512(a);
            __m512i b0 = _mm512_loadu_si512(b);
            __m512i a1 = _mm512_loadu_si512(a + 1);
            __m512i b1 = _mm512_loadu_si512(b + 1);
            __m512i a2 = _mm512_loadu_si512(a + 2);
            __m512i b2 = _mm512_loadu_si512(b + 2);
            __m512i a3 = _mm512_loadu_si512(a + 3);
            __m512i b3 = _mm512_loadu_si512(b + 3);

            // 2. Perform all XOR operations
            __m512i diff0 = _mm512_xor_si512(a0, b0);
            __m512i diff1 = _mm512_xor_si512(a1, b1);
            __m512i diff2 = _mm512_xor_si512(a2, b2);
            __m512i diff3 = _mm512_xor_si512(a3, b3);

            // 3. Perform all POPCNT operations
            __m512i p0 = _mm512_popcnt_epi64(diff0);
            __m512i p1 = _mm512_popcnt_epi64(diff1);
            __m512i p2 = _mm512_popcnt_epi64(diff2);
            __m512i p3 = _mm512_popcnt_epi64(diff3);

            // 4. Perform all addition operations
            sumv0 = _mm512_add_epi64(sumv0, p0);
            sumv1 = _mm512_add_epi64(sumv1, p1);
            sumv2 = _mm512_add_epi64(sumv2, p2);
            sumv3 = _mm512_add_epi64(sumv3, p3);

            a += 4;
            b += 4;
        }

        // Merge the results of four accumulators
        sumv0 = _mm512_add_epi64(sumv0, sumv1);
        sumv2 = _mm512_add_epi64(sumv2, sumv3);
        __m512i sumv = _mm512_add_epi64(sumv0, sumv2);

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {
            // Load 64 bytes of data into a 512 bit register
            __m512i a_vec = _mm512_loadu_si512(a);
            __m512i b_vec = _mm512_loadu_si512(b);
            __m512i diff = _mm512_xor_si512(a_vec, b_vec);
            __m512i popcnt_vec = _mm512_popcnt_epi64(diff);
            
            sumv = _mm512_add_epi64(sumv, popcnt_vec);

            // update pointers
            a += 1;
            b += 1;
        }

        for (; k + m256_SIMD_WIDTH <= K; k += m256_SIMD_WIDTH)
        {
            const __mmask8 mask = 0x0F; // 0b00001111
            __m512i zero = _mm512_setzero_si512();

            __m512i w = _mm512_mask_loadu_epi64(zero, mask, a);
            __m512i v = _mm512_mask_loadu_epi64(zero, mask, b);

            __m512i diff = _mm512_xor_si512(w, v);

            __m512i popcnt_vec = _mm512_popcnt_epi64(diff);

            sumv = _mm512_add_epi64(sumv, popcnt_vec);

            a += 1;
            b += 1;
        }

        size_t sum_diff = _mm512_reduce_add_epi64(sumv);

        return static_cast<Ret>(n_feature - sum_diff);
    }


    template <typename Ret = int>
    inline Ret bit2_cmp(const uint8_t *__restrict p1,
                       const uint8_t *__restrict p2,
                       size_t K)
    {
        constexpr size_t SIMD_WIDTH = 64;

        size_t k = 0;
        const __m512i *a = (const __m512i *)(p1);
        const __m512i *b = (const __m512i *)(p2);
        __m512i one_bits = _mm512_set1_epi64(-1);
        // 01010101 = 85
        __m512i and_mask = _mm512_set1_epi8(85);
        __m512i sumv = _mm512_setzero_si512();

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {

            __m512i a0 = _mm512_loadu_si512(a);
            __m512i b0 = _mm512_loadu_si512(b);
            // xor
            __m512i diff0 = _mm512_xor_si512(a0, b0);

            __m512i diff0_r1 = _mm512_srli_epi64(diff0, 1);

            __m512i mask = _mm512_or_si512(diff0, diff0_r1);

            __m512i unmask = _mm512_xor_si512(mask, one_bits);

            __m512i final = _mm512_and_si512(unmask, and_mask);

            __m512i popcnt_vec = _mm512_popcnt_epi64(final);
            sumv = _mm512_add_epi64(sumv, popcnt_vec);

            a += 1;
            b += 1;
        }

        long long sum = _mm512_reduce_add_epi64(sumv);
        return static_cast<Ret>(sum);
    }

    template <typename Ret = int>
    inline Ret bit2_cmp_v2(const uint8_t *__restrict p1,
                        const uint8_t *__restrict p2,
                        size_t K)
    {
        constexpr size_t SIMD_WIDTH = 64;
        size_t n_bits = K * 8;
        size_t k = 0;

        const __m512i *a = (const __m512i *)(p1);
        const __m512i *b = (const __m512i *)(p2);
        // 10101010 = 170
        __m512i or_mask = _mm512_set1_epi8(170);
        __m512i sumv = _mm512_setzero_si512();

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {

            __m512i a0 = _mm512_loadu_si512(a);
            __m512i b0 = _mm512_loadu_si512(b);
            // xor
            __m512i diff0 = _mm512_xor_si512(a0, b0);

            __m512i diff0_r1 = _mm512_srli_epi64(diff0, 1);

            __m512i mask = _mm512_or_si512(diff0, diff0_r1);

            __m512i final = _mm512_or_si512(mask, or_mask);

            __m512i popcnt_vec = _mm512_popcnt_epi64(final);
            sumv = _mm512_add_epi64(sumv, popcnt_vec);

            a += 1;
            b += 1;
        }

        size_t sum = _mm512_reduce_add_epi64(sumv);
        return static_cast<Ret>(n_bits - sum);
    }

    template <typename Ret = int>
    inline Ret bit4_cmp(const uint8_t *__restrict p1,
                           const uint8_t *__restrict p2,
                           size_t K)
    {
        constexpr size_t SIMD_WIDTH = 64;
        size_t k = 0;

        const __m512i *a = (const __m512i *)(p1);
        const __m512i *b = (const __m512i *)(p2);
        __m512i one_bits = _mm512_set1_epi64(-1);
        // 00010001 = 17
        __m512i and_mask = _mm512_set1_epi8(17);
        __m512i sumv = _mm512_setzero_si512();

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {

            __m512i a0 = _mm512_loadu_si512(a);
            __m512i b0 = _mm512_loadu_si512(b);
            // xor
            __m512i diff0 = _mm512_xor_si512(a0, b0);
            
            __m512i diff0_r1 = _mm512_srli_epi64(diff0, 1);
            __m512i diff0_r2 = _mm512_srli_epi64(diff0_r1, 1);
            __m512i diff0_r3 = _mm512_srli_epi64(diff0_r2, 1);

            // or
            __m512i mask1 = _mm512_or_si512(diff0, diff0_r1);
            __m512i mask2 = _mm512_or_si512(diff0_r2, diff0_r3);
            __m512i mask = _mm512_or_si512(mask1, mask2);

            
            __m512i unmask = _mm512_xor_si512(mask, one_bits);

            __m512i final = _mm512_and_si512(unmask, and_mask);

            __m512i popcnt_vec = _mm512_popcnt_epi64(final);
            sumv = _mm512_add_epi64(sumv, popcnt_vec);

            a += 1;
            b += 1;
        }

        long long sum = _mm512_reduce_add_epi64(sumv);
        return static_cast<Ret>(sum);
    }

    template <typename Ret = int>
    inline Ret bit4_cmp_v2(const uint8_t *__restrict p1,
                           const uint8_t *__restrict p2,
                           size_t K)
    {
        constexpr size_t SIMD_WIDTH = 64;
        size_t n_bits = K * 8;
        size_t k = 0;

        const __m512i *a = (const __m512i *)(p1);
        const __m512i *b = (const __m512i *)(p2);
        // 11101110 = 238
        __m512i or_mask = _mm512_set1_epi8(238);
        __m512i sumv = _mm512_setzero_si512();

        for (; k + SIMD_WIDTH <= K; k += SIMD_WIDTH)
        {

            __m512i a0 = _mm512_loadu_si512(a);
            __m512i b0 = _mm512_loadu_si512(b);
            // xor
            __m512i diff0 = _mm512_xor_si512(a0, b0);
            
            __m512i diff0_r1 = _mm512_srli_epi64(diff0, 1);
            __m512i diff0_r2 = _mm512_srli_epi64(diff0_r1, 1);
            __m512i diff0_r3 = _mm512_srli_epi64(diff0_r2, 1);

            // or
            __m512i mask1 = _mm512_or_si512(diff0, diff0_r1);
            __m512i mask2 = _mm512_or_si512(diff0_r2, diff0_r3);
            __m512i mask = _mm512_or_si512(mask1, mask2);

            __m512i final = _mm512_or_si512(mask, or_mask);

            __m512i popcnt_vec = _mm512_popcnt_epi64(final);
            sumv = _mm512_add_epi64(sumv, popcnt_vec);

            a += 1;
            b += 1;
        }

        size_t sum = _mm512_reduce_add_epi64(sumv);
        return static_cast<Ret>(n_bits - sum);
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

#endif // !SIM_COMPUTER_H