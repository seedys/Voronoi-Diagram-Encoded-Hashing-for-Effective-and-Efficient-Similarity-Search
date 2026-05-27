/*
iForest is the model used in the IKE method to transform dense vectors into sparse representations.
The class iForest implements this model, which is essentially an ensemble of $t$ class iTree instances.
The class IForestMap_nobias wraps the class iForest and implements the specific transform process.

IForestMap_nobias provides multiple transformation types:

1. Batch points: Transforms all input dense vectors and stores the results in an IFlib::SparseDataUINT8 structure for output.

2. Single point: Takes a single dense vector as input and outputs the transformed result as a std::vector<uint8_t>.

3. Faiss-style: Takes n dense vectors as input along with a pre-allocated memory address ``uint8_t* codes" 
    for storing the transformed data, with no return value.
*/

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <random>
#include <omp.h>
#include <memory>
#include <queue>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "IK/IK_utils.h"

namespace IFlib{

    class iTree
    {
    private:
        struct Node
        {
            bool is_leaf;
            int split_feature;
            float split_value;
            Node *left;
            Node *right;
            uint8_t leaf_index;

            Node(int feat, float val, Node *l, Node *r)
                : is_leaf(false), split_feature(feat),
                split_value(val), left(l), right(r), leaf_index(255) {}

            Node(int index)
                : is_leaf(true), split_feature(-1),
                split_value(0.0f), left(nullptr), right(nullptr), leaf_index(index) {}
        };

        Node *root;
        int max_depth;
        size_t _d;                                 // Original data dimension
        const std::vector<const float *> &data_ref; // Reference to raw corpus data

        Node *build_tree(const std::vector<size_t> &indices, int depth, int &leaf_counter, std::mt19937 &engine)
        {
            if (indices.size() <= 1 || depth >= max_depth)
            {
                return new Node(leaf_counter++);
            }

            // feature selection
            std::uniform_int_distribution<size_t> feat_dist(0, _d - 1);
            int feat = static_cast<int>(feat_dist(engine));

            // Calculate feature range
            float min_val, max_val;
            min_val = data_ref[indices[0]][feat];
            max_val = data_ref[indices[0]][feat];
            for (size_t idx : indices)
            {
                min_val = std::min(min_val, data_ref[idx][feat]);
                max_val = std::max(max_val, data_ref[idx][feat]);
            }

            std::uniform_real_distribution<float> split_dist(0.0f, 1.0f);
            float split = min_val + split_dist(engine) * (max_val - min_val);

            // split indices
            std::vector<size_t> left_indices, right_indices;
            for (size_t idx : indices)
            {
                (data_ref[idx][feat] < split ? left_indices : right_indices).push_back(idx);
            }

            if (left_indices.empty() || right_indices.empty())
            {
                return new Node(leaf_counter++);
            }

            return new Node(feat, split,
                            build_tree(left_indices, depth + 1, leaf_counter, engine),
                            build_tree(right_indices, depth + 1, leaf_counter, engine));
        }

    public:
        iTree(const std::vector<const float*> &data, const std::vector<size_t> &indices, 
            size_t psi, size_t d, std::mt19937 &engine)
            : data_ref(data), _d(d)
        {
            int leaf_counter = 0;
            max_depth = static_cast<int>(ceil(log2(psi)));
            root = build_tree(indices, 0, leaf_counter, engine);
        }

        uint8_t apply(const float* sample) const
        {
            Node *current = root;
            while (!current->is_leaf)
            {
                current = (sample[current->split_feature] < current->split_value)
                            ? current->left
                            : current->right;
            }
            return current->leaf_index;
        }
    };

    class iForest
    {
    private:
        size_t _psi;
        size_t _t;
        size_t _d;
        std::vector<std::unique_ptr<iTree>> trees;
        const std::vector<const float*> &data_ref;
        bool use_without_replacement;

    public:
        iForest(const std::vector<const float*> &data, size_t psi, size_t t, 
            size_t d, int seed, bool use_without_replacement)
            : data_ref(data), _psi(psi), _t(t), _d(d), use_without_replacement(use_without_replacement)
        {
            // Check if sampling without replacement is feasible
            const size_t N = data_ref.size();
            if (use_without_replacement && N < _t * _psi)
            {
                std::cerr << "Warning: Insufficient sample size, automatically switching to sampling with replacement mode" << std::endl;
                use_without_replacement = false;
            }

            // Pre generate a random engine for each iTree
            std::vector<std::mt19937> engines;
            std::mt19937 seed_gen(seed);
            std::uniform_int_distribution<int> seed_dist;
            for (size_t i = 0; i < t; ++i)
            {
                engines.emplace_back(seed_dist(seed_gen));
            }

            // If there is no replacement sampling, first extract _t * _psi globally unique indexes at once
            std::vector<size_t> global_indices;
            if (use_without_replacement)
            {
                std::vector<size_t> pool(N);
                std::iota(pool.begin(), pool.end(), 0);

                std::shuffle(pool.begin(), pool.end(), engines[0]);

                // take the first _t * _psi
                global_indices.assign(pool.begin(), pool.begin() + (_t * _psi));
            }

            // Start generating tree
            trees.resize(_t);
    #pragma omp parallel for
            for (size_t i = 0; i < _t; ++i)
            {
                std::vector<size_t> indices(_psi);

                if (use_without_replacement)
                {
                    size_t offset = i * _psi;
                    for (size_t j = 0; j < _psi; ++j)
                    {
                        indices[j] = global_indices[offset + j];
                    }
                }
                else
                {
                    std::uniform_int_distribution<size_t> index_dist(0, N - 1);
                    auto &engine = engines[i];
                    for (size_t j = 0; j < _psi; ++j)
                    {
                        indices[j] = index_dist(engine);
                    }
                }

                trees[i] = std::make_unique<iTree>(data_ref, indices, _psi, _d, engines[i]);
            }
        }

        std::vector<uint8_t> apply(const float* sample) const
        {
            std::vector<uint8_t> path(_t);
            for (size_t i = 0; i < _t; i++)
            {
                path[i] = trees[i]->apply(sample);
            }
            return path;
        }
    };

    class IForestMap_nobias
    {
    private:
        iForest forest;
        size_t _t;
        size_t _psi;
        size_t _d;

    public:
        IForestMap_nobias(const std::vector<const float*> &data_ptr, size_t t, size_t psi,
                        size_t d, bool use_without_replacement, int seed = 0)
            : _t(t), _psi(psi), _d(d), forest(data_ptr, psi, t, d, seed, use_without_replacement)
        {
            if (psi > 256)
            {
                throw std::invalid_argument("psi value cannot exceed 256, current value: " + std::to_string(psi));
            }
        }

        /*================================== Batch points transform function ==================================*/

        SparseDataUINT8 transform(const std::vector<std::vector<float>> &X) const
        {
            SparseDataUINT8 result;
            result.n_features = _t;
            result.n_samples = X.size();

            // Convert to continuous memory layout
            result.data_flat.resize(result.n_features * result.n_samples);
            result.data_ptrs.resize(result.n_samples);

    #pragma omp parallel for
            for (size_t i = 0; i < X.size(); ++i)
            {
                auto path = forest.apply(X[i].data());
                size_t offset = i * result.n_features;

                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    result.data_flat[offset + tree_idx] = path[tree_idx];
                }

                result.data_ptrs[i] = result.data_flat.data() + offset;
            }
            return result;
        }

        SparseDataUINT8 transform_packed(const std::vector<std::vector<float>> &X) const
        {
            if (_psi != 2)
            {
                throw std::invalid_argument("transform_packed function requires psi value to be 2, current value: " + std::to_string(_psi));
            }

            SparseDataUINT8 result;
            result.n_samples = X.size();

            // code legth in bytes = ceil(n_features / 8)
            size_t bytes_per_sample = (_t + 7) / 8;
            result.n_features = bytes_per_sample;

            result.data_flat.resize(result.n_samples * bytes_per_sample);
            result.data_ptrs.resize(result.n_samples);

    #pragma omp parallel for
            for (size_t i = 0; i < X.size(); ++i)
            {
                const auto path = forest.apply(X[i].data()); // return vector<uint8_t>, with a value of 0 or 1
                size_t base = i * bytes_per_sample;
                
                std::fill_n(result.data_flat.data() + base, bytes_per_sample, 0u);

                // packed
                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    if (path[tree_idx])
                    {
                        size_t byte_idx = base + (tree_idx >> 3); // tree_idx / 8
                        size_t bit_idx = tree_idx & 7;            // tree_idx % 8
                        result.data_flat[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
                    }
                }
                result.data_ptrs[i] = result.data_flat.data() + base;
            }

            return result;
        }

        SparseDataUINT8 transform_packed_2bit(const std::vector<std::vector<float>> &X) const
        {
            if (_psi > 4 || _psi < 3)
            {
                throw std::invalid_argument("transform_packed_2bit function requires psi to be in the range [3, 4], current value: " + std::to_string(_psi));
            }

            SparseDataUINT8 result;
            result.n_samples = X.size();

            // code length in bytes = ceil(n_features * 2 / 8) = ceil(n_features / 4)
            size_t bytes_per_sample = (_t + 3) / 4;
            result.n_features = bytes_per_sample; 

            result.data_flat.resize(result.n_samples * bytes_per_sample);
            result.data_ptrs.resize(result.n_samples);

    #pragma omp parallel for
            for (size_t i = 0; i < X.size(); ++i)
            {
                const auto path = forest.apply(X[i].data()); // Return vector<uint8_t>, with a value of 0, 1, 2 or 3
                size_t base = i * bytes_per_sample;
                
                std::fill_n(result.data_flat.data() + base, bytes_per_sample, 0u);

                // packed (2 bits)
                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    uint8_t value = path[tree_idx] & 0x03;

                    size_t byte_idx = base + (tree_idx >> 2);  // tree_idx / 4
                    size_t bit_offset = (tree_idx & 0x03) * 2; // (tree_idx % 4) * 2

                    result.data_flat[byte_idx] |= static_cast<uint8_t>(value << (6 - bit_offset));
                }
                result.data_ptrs[i] = result.data_flat.data() + base;
            }

            return result;
        }

        SparseDataUINT8 transform_packed_4bit(const std::vector<std::vector<float>> &X) const
        {
            if (_psi > 16)
            {
                throw std::invalid_argument("transform_packed_4bit function requires psi ≤ 16, current value: " + std::to_string(_psi));
            }

            SparseDataUINT8 result;
            result.n_samples = X.size();

            // code size in bytes = ceil(n_features * 4 / 8) = ceil(n_features / 2)
            size_t bytes_per_sample = (_t + 1) / 2;
            result.n_features = bytes_per_sample;

            result.data_flat.resize(result.n_samples * bytes_per_sample);
            result.data_ptrs.resize(result.n_samples);

            
    #pragma omp parallel for
            for (size_t i = 0; i < X.size(); ++i)
            {
                const auto path = forest.apply(X[i].data()); // vector<uint8_t>，value of 0-15
                size_t base = i * bytes_per_sample;
                
                std::fill_n(result.data_flat.data() + base, bytes_per_sample, 0u);

                // packed (4 bits)
                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    uint8_t value = path[tree_idx] & 0x0F;

                    size_t byte_idx = base + (tree_idx >> 1);  // tree_idx / 2
                    size_t bit_offset = (tree_idx & 0x01) * 4; // (tree_idx % 2) * 4

                    result.data_flat[byte_idx] |= static_cast<uint8_t>(value << (4 - bit_offset));
                }
                result.data_ptrs[i] = result.data_flat.data() + base;
            }

            return result;
        }

        /*================================== Single point transform function ==================================*/

        std::vector<uint8_t> transform_single_point(const float* x) const
        {
            return forest.apply(x);
        }
        
        std::vector<uint8_t> transform_single_point_packed(const float *x) const
        {
            size_t bytes_per_sample = (_t + 7) / 8;
            std::vector<uint8_t> res(bytes_per_sample, 0);

            const auto path = forest.apply(x);

            for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
            {
                if (path[tree_idx])
                {
                    size_t byte_idx = tree_idx >> 3; // tree_idx / 8
                    size_t bit_idx = tree_idx & 7;            // tree_idx % 8
                    res[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
                }
            }

            return res;
        }

        std::vector<uint8_t> transform_single_point_packed_2bit(const float *x) const
        {

            size_t bytes_per_sample = (_t + 3) / 4;
            std::vector<uint8_t> res(bytes_per_sample, 0);

            const auto path = forest.apply(x);

            // packed 2 bits
            for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
            {
                uint8_t value = path[tree_idx] & 0x03;

                size_t byte_idx = tree_idx >> 2;           // tree_idx / 4
                size_t bit_offset = (tree_idx & 0x03) * 2; // (tree_idx % 4) * 2

                res[byte_idx] |= static_cast<uint8_t>(value << (6 - bit_offset));
            }

            return res;
        }

        std::vector<uint8_t> transform_single_point_packed_4bit(const float *x) const
        {

            size_t bytes_per_sample = (_t + 1) / 2;
            std::vector<uint8_t> res(bytes_per_sample, 0);

            const auto path = forest.apply(x);

            // packed 4 bits
            for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
            {
                uint8_t value = path[tree_idx] & 0x0F;

                size_t byte_idx = tree_idx >> 1;           // tree_idx / 2
                size_t bit_offset = (tree_idx & 0x01) * 4; // (tree_idx % 2) * 4

                res[byte_idx] |= static_cast<uint8_t>(value << (4 - bit_offset));
            }

            return res;
        }

        /*============================================ Faiss-style function ============================================*/
        void transform(size_t n, const float *x, uint8_t *codes) const
        {
    #pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                auto path = forest.apply(x + i * _d);
                size_t offset = i * _t;

                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    codes[offset + tree_idx] = path[tree_idx];
                }

            }
        }

        void transform_packed(size_t n, const float *x, uint8_t *codes) const
        {
            if (_psi != 2)
            {
                throw std::invalid_argument("transform_packed function requires psi value to be 2, current value: " + std::to_string(_psi));
            }

            // code length in bytes = ceil(n_features / 8)
            size_t bytes_per_sample = (_t + 7) / 8;

    #pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                auto path = forest.apply(x + i * _d);
                size_t base = i * bytes_per_sample;
                
                std::fill_n(codes + base, bytes_per_sample, 0u);

                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    if (path[tree_idx])
                    {
                        size_t byte_idx = base + (tree_idx >> 3); // tree_idx / 8
                        size_t bit_idx = tree_idx & 7;            // tree_idx % 8
                        codes[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
                    }
                }
            }
            
        }

        void transform_packed_2bit(size_t n, const float *x, uint8_t *codes) const
        {
            if (_psi > 4 || _psi < 3)
            {
                throw std::invalid_argument("transform_packed_2bit function requires psi to be in the range [3, 4], current value: " + std::to_string(_psi));
            }

            size_t bytes_per_sample = (_t + 3) / 4;

#pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                const auto path = forest.apply(x + i * _d);
                size_t base = i * bytes_per_sample;
            
                std::fill_n(codes + base, bytes_per_sample, 0u);

                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    uint8_t value = path[tree_idx] & 0x03;

                    size_t byte_idx = base + (tree_idx >> 2);  // tree_idx / 4
                    size_t bit_offset = (tree_idx & 0x03) * 2; // (tree_idx % 4) * 2

                    codes[byte_idx] |= static_cast<uint8_t>(value << (6 - bit_offset));
                }
            }
        }

        void transform_packed_4bit(size_t n, const float *x, uint8_t *codes) const
        {
            if (_psi > 16)
            {
                throw std::invalid_argument("transform_packed_4bit function requires psi ≤ 16, current value: " + std::to_string(_psi));
            }

            size_t bytes_per_sample = (_t + 1) / 2;

#pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                const auto path = forest.apply(x + i * _d);
                size_t base = i * bytes_per_sample;

                std::fill_n(codes + base, bytes_per_sample, 0u);

                // packed (4 bits)
                for (size_t tree_idx = 0; tree_idx < _t; ++tree_idx)
                {
                    uint8_t value = path[tree_idx] & 0x0F;

                    size_t byte_idx = base + (tree_idx >> 1);  // tree_idx / 2
                    size_t bit_offset = (tree_idx & 0x01) * 4; // (tree_idx % 2) * 4

                    codes[byte_idx] |= static_cast<uint8_t>(value << (4 - bit_offset));
                }
            }
        }
    };

} // namespace IFlib
