#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include <random>
#include <stdexcept>
#include <algorithm>

namespace IFlib{
    struct RandomGenerator
    {
    std::mt19937 engine;
    std::uniform_real_distribution<float> float_dist;

    explicit RandomGenerator(long long seed = 1234)
        : engine(seed), float_dist(0.0f, 1.0f) {}

    float rand_float()
    {
        return float_dist(engine);
    }

    int rand_int(int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("RandomGenerator::rand_int: n must be positive");
        }
        std::uniform_int_distribution<int> int_dist(0, n - 1);
        return int_dist(engine);
    }

    std::vector<size_t> select_random_indices(size_t _n_base_data, size_t k)
    {
        std::vector<size_t> indices;

        // Processing invalid inputs
        if (k == 0 || _n_base_data == 0 || k > _n_base_data)
        {
            return indices;
        }

        // idx list: 0, 1, 2, ..., _n_base_data-1
        indices.reserve(_n_base_data);
        for (size_t i = 0; i < _n_base_data; ++i)
        {
            indices.push_back(i);
        }

        // shuffle
        std::shuffle(indices.begin(), indices.end(), engine);

        // Extract the first k elements
        indices.resize(k);

        return indices;
    }

    };
} // namespace IFlib

#endif // RANDOM_GENERATOR_H