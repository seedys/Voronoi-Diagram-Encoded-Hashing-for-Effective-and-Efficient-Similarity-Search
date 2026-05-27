#include "IForest_index/IForest_Flat.h"

namespace IFlib{
    void IForestFlat::init_functions()
    {
        if (psi == 2)
        {
            transform_func = &IForestMap_nobias::transform_packed;
            gemc_simd_func = gemc_simd_bits;
        }
        else if (psi <= 4)
        {
            transform_func = &IForestMap_nobias::transform_packed_2bit;
            gemc_simd_func = gemc_simd_2bits;
        }
        else if (psi <= 16)
        {
            transform_func = &IForestMap_nobias::transform_packed_4bit;
            gemc_simd_func = gemc_simd_4bits;
        }
        else
        {
            transform_func = &IForestMap_nobias::transform;
            gemc_simd_func = gemc_simd_parallel_1x4;
        }
    }

    IForestFlat::IForestFlat(std::vector<std::vector<float>> &base_data, size_t t, size_t psi,
                             bool use_without_replacement, int seed)
        : psi(psi), t(t)
    {
        init_functions();
        ntotal = base_data.size();
        d = base_data[0].size();
        std::vector<const float *> data_ptr(ntotal);
        for (size_t i = 0; i < ntotal; i++)
            data_ptr[i] = base_data[i].data();

        // train model
        model = std::make_unique<IForestMap_nobias>(data_ptr, t, psi, d, use_without_replacement, seed);
        
        // add data
        SparseDataUINT8 raw_base_data = (model.get()->*transform_func)(base_data);
        encoded_data = std::move(raw_base_data.data_flat);
        encoded_data_ptr = std::move(raw_base_data.data_ptrs);

        // set code_size
        code_size = raw_base_data.n_features;
    }

    std::vector<std::vector<size_t>> IForestFlat::query(
        std::vector<std::vector<float>> &query_data, 
        size_t k)
    {
        size_t n_queries = query_data.size();

        SparseDataUINT8 raw_query_data = (model.get()->*transform_func)(query_data);
        const uint8_t *xq = raw_query_data.data_ptrs[0];

        IFlib::BatchMinHeapHandler res_handler(n_queries, k);
        const size_t bs_x = 4096;
        const size_t bs_y = 1024;

        std::chrono::duration<double> total_gemc_duration = std::chrono::duration<double>::zero();
        for (size_t i0 = 0; i0 < n_queries; i0 += bs_x)
        {
            size_t i1 = std::min(n_queries, i0 + bs_x);

            for (size_t j0 = 0; j0 < ntotal; j0 += bs_y)
            {
                size_t j1 = std::min(ntotal, j0 + bs_y);

                {
                    size_t M = i1 - i0, N = j1 - j0;

                    gemc_simd_func(xq + i0 * code_size,
                                   encoded_data_ptr[j0],
                                   M, N, code_size, i0, j0, res_handler);

                }
            }
        }

        return res_handler.sort_all();
    }

} // namespace IFlib