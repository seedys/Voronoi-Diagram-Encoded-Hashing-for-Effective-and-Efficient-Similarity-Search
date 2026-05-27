#pragma once

/*
Create a iForest model and conduct Exhaustive Search
*/

#include <vector>
#include <chrono>

#include "IK/iForest.cpp"
#include "IForest_index/gemc.h"
#include "tools/ResHandler.h"

namespace IFlib
{
    struct IForestFlat
    {
    private:
        size_t psi;
        size_t t;
        size_t d; // original data dimension

        size_t code_size;
        size_t ntotal;

        std::vector<uint8_t, aligned_allocator<uint8_t, 64>> encoded_data;
        std::vector<uint8_t *> encoded_data_ptr;

        std::unique_ptr<IForestMap_nobias> model; // IF model

        // function pointer
        typedef void (*GemcSimdFunc)(const uint8_t *, const uint8_t *, size_t, size_t, size_t, size_t, size_t, BatchMinHeapHandler &);
        GemcSimdFunc gemc_simd_func;

        typedef SparseDataUINT8 (IForestMap_nobias::*TransformFunc)(const std::vector<std::vector<float>> &) const;
        TransformFunc transform_func;

        void init_functions();

    public:
        IForestFlat(std::vector<std::vector<float>> &base_data, size_t t, size_t psi,
                    bool use_without_replacement = false, int seed = 0);

        std::vector<std::vector<size_t>> query(
            std::vector<std::vector<float>> &query_data,
            size_t k);
    };

} // namespace IFlib