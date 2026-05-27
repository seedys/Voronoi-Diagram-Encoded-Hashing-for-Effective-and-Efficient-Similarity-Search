#include "utils.h"

vector<vector<int32_t>> read_ivecs(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
        throw runtime_error("Unable to open file: " + filename);

    vector<vector<int32_t>> data;

    while (true)
    {
        // Read dimension (small end int32)
        int32_t dim;
        in.read(reinterpret_cast<char *>(&dim), 4);
        if (in.gcount() != 4)
            break; // done

        if (dim <= 0)
            throw runtime_error("Invalid dimension: " + to_string(dim));

        // Read data block (small end int32 array)
        vector<int32_t> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(int32_t));
        if (in.gcount() != dim * sizeof(int32_t))
            throw runtime_error("Incomplete file data");

        data.push_back(move(vec));
    }

    return data;
}

vector<vector<float>> read_fvecs(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
        throw runtime_error("Unable to open file: " + filename);

    vector<vector<float>> data;

    while (true)
    {
        // small end int32
        int32_t dim;
        in.read(reinterpret_cast<char *>(&dim), 4);
        if (in.gcount() != 4)
            break;

        if (dim <= 0)
            throw runtime_error("Invalid dimension: " + to_string(dim));

        // read chunk
        vector<float> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (in.gcount() != dim * sizeof(float))
            throw runtime_error("Incomplete file data");

        data.push_back(move(vec));
    }

    return data;
}

float *read_fvecs(const string &filename, size_t &n, size_t &d)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);

    // Read first dimension (little-endian int32)
    int32_t dim;
    in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    if (in.gcount() != sizeof(dim))
        throw std::runtime_error("Failed to read dimension from file: " + filename);
    if (dim <= 0)
        throw std::runtime_error("Invalid dimension: " + std::to_string(dim));
    d = static_cast<size_t>(dim);

    // Determine file size
    auto cur_pos = in.tellg();
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    in.seekg(cur_pos);

    const size_t header_bytes = sizeof(int32_t);
    const size_t vector_bytes = d * sizeof(float);
    const size_t record_size = header_bytes + vector_bytes;

    if ((file_size % record_size) != 0)
        throw std::runtime_error("File size is not a multiple of record size");

    n = static_cast<size_t>(file_size / record_size);

    // Allocate contiguous memory for all vectors
    float *data = new float[n * d];

    // Rewind to beginning and load
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i)
    {
        int32_t cur_dim;
        in.read(reinterpret_cast<char *>(&cur_dim), sizeof(cur_dim));
        if (in.gcount() != sizeof(cur_dim))
            throw std::runtime_error("Unexpected EOF reading dimension at index " + std::to_string(i));
        if (cur_dim != dim)
            throw std::runtime_error("Inconsistent dimension at index " + std::to_string(i));

        // Read vector directly into contiguous block
        float *vec_ptr = data + i * d;
        in.read(reinterpret_cast<char *>(vec_ptr), vector_bytes);
        if (static_cast<size_t>(in.gcount()) != vector_bytes)
            throw std::runtime_error("Unexpected EOF reading vector at index " + std::to_string(i));
    }

    return data;
}

vector<vector<uint32_t>> read_uivecs(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
        throw runtime_error("could not open file: " + filename);

    vector<vector<uint32_t>> data;
    int32_t dim;

    while (in.read(reinterpret_cast<char *>(&dim), sizeof(int32_t)))
    {
        if (dim <= 0)
            throw runtime_error("Invalid dimension: " + to_string(dim));

        vector<uint32_t> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(uint32_t));
        if (in.gcount() != dim * sizeof(uint32_t))
            throw runtime_error("Incomplete file data");

        data.push_back(move(vec));
    }

    return data;
}

double compute_mrr_at_r(const std::vector<std::vector<uint32_t>> &labels,
                   const std::vector<std::vector<size_t>> &query_res,
                   size_t R)
{
    const size_t batch_size = labels.size();
    if (query_res.size() != batch_size)
    {
        throw std::invalid_argument("Labels and query results batch size mismatch");
    }

    if (R == 0)
    {
        throw std::invalid_argument("R must be greater than 0");
    }

    double total_mrr = 0.0;
    size_t valid_queries = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
        // Got all groundtruths of the current query
        const auto &current_labels = labels[i];

        // Skip invalid queries without ground truth
        if (current_labels.empty())
        {
            continue;
        }

        // groundtruth set for current query
        std::unordered_set<uint32_t> gt_set(current_labels.begin(), current_labels.end());

        // Traverse the first R results to find the first hit
        bool found = false;
        size_t first_rank = 0;
        const size_t actual_R = std::min(R, query_res[i].size());

        for (size_t j = 0; j < actual_R; ++j)
        {
            if (gt_set.count(query_res[i][j]))
            {
                first_rank = j + 1;
                found = true;
                break;
            }
        }

        if (found)
        {
            total_mrr += 1.0 / first_rank;
        }

        ++valid_queries;
    }

    if (valid_queries == 0)
    {
        throw std::runtime_error("No valid queries with groundtruth");
    }

    return total_mrr / valid_queries;
}

double compute_ndcg_at_r(const std::vector<std::vector<uint32_t>> &labels,
                    const std::vector<std::vector<size_t>> &query_res,
                    size_t R,
                    const std::vector<std::vector<uint32_t>> &relevance_scores)
{
    const size_t batch_size = labels.size();
    if (query_res.size() != batch_size)
    {
        throw std::invalid_argument("Labels and query results batch size mismatch");
    }

    bool use_multilevel = !relevance_scores.empty();
    if (use_multilevel && relevance_scores.size() != batch_size)
    {
        throw std::invalid_argument("relevance_scores must have same batch size as labels");
    }

    if (R == 0)
    {
        throw std::invalid_argument("R must be greater than 0");
    }

    double total_ndcg = 0.0;
    size_t valid_queries = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
        const auto &current_labels = labels[i];
        if (current_labels.empty())
        {
            continue;
        }

        // Verify the correlation score dimension
        if (use_multilevel && relevance_scores[i].size() != current_labels.size())
        {
            throw std::invalid_argument("relevance_scores[" + std::to_string(i) + "] size mismatch");
        }

        // Build a mapping from tags to scores
        std::unordered_map<uint32_t, int> label_score_map;
        std::vector<int> gt_scores;
        for (size_t j = 0; j < current_labels.size(); ++j)
        {
            int score = use_multilevel ? relevance_scores[i][j] : 1;
            label_score_map[current_labels[j]] = score;
            gt_scores.push_back(score);
        }

        // compute DCG@R
        double dcg = 0.0;
        const size_t actual_R = std::min(R, query_res[i].size());
        for (size_t j = 0; j < actual_R; ++j)
        {
            const auto &item = query_res[i][j];
            auto it = label_score_map.find(item);
            if (it != label_score_map.end())
            {
                dcg += it->second / std::log2(j + 2);
            }
        }

        // compute IDCG@R
        std::sort(gt_scores.rbegin(), gt_scores.rend()); // descending order sort
        std::vector<int> ideal_scores(gt_scores.begin(), gt_scores.begin() + std::min(R, gt_scores.size()));
        ideal_scores.resize(R, 0); // Fill in 0 until R elements

        double idcg = 0.0;
        for (size_t j = 0; j < R; ++j)
        {
            idcg += ideal_scores[j] / std::log2(j + 2);
        }

        // compute nDCG
        if (idcg > 0)
        {
            total_ndcg += dcg / idcg;
        }

        ++valid_queries;
    }

    if (valid_queries == 0)
    {
        throw std::runtime_error("No valid queries with groundtruth");
    }

    return total_ndcg / valid_queries;
}


std::vector<size_t> select_random_indices(size_t _n_base_data, size_t k)
{
    std::vector<size_t> indices;

    if (k == 0 || _n_base_data == 0 || k > _n_base_data)
    {
        return indices;
    }

    indices.reserve(_n_base_data);
    for (size_t i = 0; i < _n_base_data; ++i)
    {
        indices.push_back(i);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(indices.begin(), indices.end(), gen);

    indices.resize(k);

    return indices;
}

void l2_normalize_rows(std::vector<std::vector<float>> &data)
{
    for (auto &row : data)
    {
        // compute L2 norm
        float sum_sq = 0.0f;
        for (float v : row)
        {
            sum_sq += v * v;
        }
        float norm = std::sqrt(sum_sq);
        
        if (norm > 0.0f)
        {
            for (auto &v : row)
            {
                v /= norm;
            }
        }
    }
}


bool fileExists(const std::string &filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}
