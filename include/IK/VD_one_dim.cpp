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
#include "IK/IK_utils.h"

using std::string;
using std::vector;

namespace IFlib
{
    class single_VD
    {
    private:
        const float *data_ref; // 原始数据引用
        const vector<size_t> _indices;
        vector<float> split_data;
        size_t split_feat;

    public:
        single_VD(const float *data, vector<size_t> &&indices,
                  const size_t psi, const size_t d, mt19937 &engine)
            : data_ref(data), _indices(move(indices))
        {
            split_data.resize(psi);

            // 特征选择
            std::uniform_int_distribution<size_t> feat_dist(0, d - 1);
            split_feat = feat_dist(engine);

            for (size_t i = 0; i < psi; ++i)
            {
                split_data[i] = data_ref[_indices[i] * d + split_feat];
            }
        }

        uint8_t apply(const float *sample)
        {
            float attri = sample[split_feat];

            auto it = std::min_element(
                split_data.begin(),
                split_data.end(),
                [&](float a, float b)
                {
                    return std::abs(a - attri) < std::abs(b - attri);
                });

            size_t idx = std::distance(split_data.begin(), it);
            return static_cast<uint8_t>(idx);
        }
    };

    class VDMap_nobias
    {
    private:
        size_t _t;
        size_t _psi;
        size_t _d;
        vector<std::unique_ptr<single_VD>> vds;

    public:
        VDMap_nobias(const float *data, size_t n, size_t d, size_t t, size_t psi, int seed = 0)
            : _t(t), _psi(psi), _d(d)
        {
            if (psi > 256)
            {
                throw std::invalid_argument("psi 值不能超过 256，当前值: " + std::to_string(psi));
            }

            // 为每棵树预生成随机引擎
            vector<std::mt19937> engines;
            std::mt19937 seed_gen(seed);
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

            vds[i] = std::make_unique<single_VD>(data, move(indices), _psi, d, engine);
        }
    }

    IFlib::SparseDataUINT8 transform(size_t n, const float *x) const
    {
        IFlib::SparseDataUINT8 result;
        result.n_features = _t;
        result.n_samples = n;

        // 转换为连续内存布局
        result.data_flat.resize(result.n_features * result.n_samples);
        result.data_ptrs.resize(result.n_samples);

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
            const float *q = x + i * _d;
            for (size_t j = 0; j < _t; j++){
                result.data_flat[i * _t + j] = vds[j]->apply(q);
            }
        }

        for (size_t data_idx = 0; data_idx < n; ++data_idx)
            result.data_ptrs[data_idx] = result.data_flat.data() + data_idx * result.n_features;

        return result;
    }
};

} // namespace IFlib