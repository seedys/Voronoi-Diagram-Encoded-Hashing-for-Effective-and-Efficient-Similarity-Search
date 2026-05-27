#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

using namespace std;


vector<vector<int32_t>> read_ivecs(const string &filename);

vector<vector<float>> read_fvecs(const string &filename);

float *read_fvecs(const string &filename, size_t &n, size_t &d);

vector<vector<uint32_t>> read_uivecs(const string &filename);

double compute_mrr_at_r(const vector<vector<uint32_t>> &labels, 
                    const vector<vector<size_t>> &query_res, 
                    size_t R);

double compute_ndcg_at_r(const std::vector<std::vector<uint32_t>> &labels,
                    const std::vector<std::vector<size_t>> &query_res,
                    size_t R,
                    const std::vector<std::vector<uint32_t>> &relevance_scores = {});

vector<size_t> select_random_indices(size_t _n_base_data, size_t k);

void l2_normalize_rows(std::vector<std::vector<float>> &data);

bool fileExists(const std::string &filename);

#endif // !UTILS_H
