#pragma once

#include "IForest_index/myHNSW.h"
#include "IK/iForest.cpp"

namespace IFlib
{
    struct IForestHNSW
    {
        typedef HNSW::storage_idx_t storage_idx_t;

        // the link structure
        HNSW hnsw;

        IForestMap_nobias *model;

        // the sequential storage
        size_t ntotal = 0;
        size_t d; // code size
        uint8_t *base_data;
        std::vector<uint8_t> encoded_base_data;

        // para. of IKE
        size_t t;
        size_t psi;

        bool verbose;

        // When set to true, all neighbors in level 0 are filled up
        // to the maximum size allowed (2 * M).
        bool keep_max_size_level0 = false;

        explicit IForestHNSW(size_t t, size_t psi, int M = 32, bool verbose = true);

        void add(size_t n, const float* x);

        /// entry point for search
        std::vector<std::vector<size_t>> search(
            size_t n,
            const float *xq,
            size_t d_,
            size_t k) const;

        void reset();
    };

} // namespace IFlib