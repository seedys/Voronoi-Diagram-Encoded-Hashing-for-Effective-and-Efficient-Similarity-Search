#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <random>
#include <omp.h>
#include <memory>
#include <queue>

#include <iostream>
#include <faiss/IndexFlat.h>

#include "IK/IK_utils.h"

using std::mt19937;
using std::string;
using std::unique_ptr;
using std::vector;

namespace IFlib
{
    class single_VD
    {
    private:
        const float *data_ref; // 原始数据引用
        const vector<size_t> _indices;

    public:
        faiss::IndexFlatL2 index;
        single_VD(const float *data, vector<size_t> &&indices, const size_t psi, const size_t d)
            : data_ref(data), _indices(move(indices)), index(d)
        {
            vector<float> tmp_data(psi * d);

            for (size_t i = 0; i < psi; ++i)
            {
                size_t idx = _indices[i];

                std::memcpy(tmp_data.data() + d * i,
                            data_ref + d * idx,
                            d * sizeof(float));
            }

            // 添加数据到Faiss索引
            index.add(psi, tmp_data.data());
        }
    };

    class VDMap_nobias
    {
    private:
        size_t _t;
        size_t _psi;
        vector<unique_ptr<single_VD>> vds;

    public:
        VDMap_nobias(const float *data, size_t n, size_t d, size_t t, size_t psi, int seed = 0)
            : _t(t), _psi(psi)
        {
            if (psi > 256)
            {
                throw std::invalid_argument("psi 值不能超过 256，当前值: " + std::to_string(psi));
            }

            // 为每棵树预生成随机引擎
            vector<mt19937> engines;
            mt19937 seed_gen(seed);
            std::uniform_int_distribution<int> seed_dist;
            for (size_t i = 0; i < t; ++i)
            {
                engines.emplace_back(seed_dist(seed_gen));
            }

            vds.resize(_t);
    #pragma omp parallel for
            for (size_t i = 0; i < _t; ++i)
            {
                auto &engine = engines[i];

                // 抽样
                std::uniform_int_distribution<size_t> index_dist(0, n - 1);
                vector<size_t> indices(_psi);
                for (auto &idx : indices)
                {
                    idx = index_dist(engine);
                }

                vds[i] = std::make_unique<single_VD>(data, std::move(indices), _psi, d); // 使用 move(indices) 转移数据所有权
            }
        }

        SparseDataUINT8 transform(const float *X, const size_t n) const
        {
            SparseDataUINT8 result;
            result.n_features = _t;
            result.n_samples = n;

            // 转换为连续内存布局
            result.data_flat.resize(result.n_features * result.n_samples);
            result.data_ptrs.resize(result.n_samples);

            for (size_t i = 0; i < _t; ++i)
            {
                std::vector<float> D(n);
                std::vector<long> I(n);

                (vds[i]->index).search(n, X, 1, D.data(), I.data());

    #pragma omp parallel for
                for (size_t data_idx = 0; data_idx < n; ++data_idx)
                {
                    // 无偏移量
                    result.data_flat[data_idx * _t + i] = static_cast<uint8_t>(I[data_idx]);
                }
                printf("%d / %d       \r",
                    i, _t);

                fflush(stdout);
            }
            printf("\n");

            for (size_t data_idx = 0; data_idx < n; ++data_idx)
                result.data_ptrs[data_idx] = result.data_flat.data() + data_idx * _t;

            return result;
        }

        SparseDataUINT8 transform_with_cache(const float *X, const size_t n, const std::string cache_path) const
        {
            SparseDataUINT8 result;
            if (!fileExists(cache_path))
            {
                result = transform(X, n);
                saveVectorToBinaryFile(result, cache_path);
            }
            else
            {
                result.n_features = _t;
                result.n_samples = n;
                result.data_flat.resize(result.n_features * result.n_samples);
                loadVectorFromBinaryFile(result, cache_path);

                result.data_ptrs.resize(result.n_samples);
                for (size_t data_idx = 0; data_idx < n; ++data_idx)
                    result.data_ptrs[data_idx] = result.data_flat.data() + data_idx * _t;
                    
                std::cout << "Load base data from " << cache_path << "\n";
            }
            return result;
        }
    };

} // namespace IFlib