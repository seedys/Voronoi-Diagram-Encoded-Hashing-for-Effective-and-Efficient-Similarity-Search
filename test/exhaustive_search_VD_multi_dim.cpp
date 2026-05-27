#include <vector>

#include <omp.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <memory>
#include <immintrin.h>
#include <cstdint>
#include <queue>

#include <nlohmann/json.hpp>
#include <tools/ResHandler.h>
#include <IForest_index/gemc.h>

#include "IK/IK_utils.h"
#include "utils.h"
#include "IK/VD_multi_dim.cpp"

using json = nlohmann::json;
using namespace std;

class Exhaustive_search_IForest_Processor
{
private:
    int _psi;
    int _t;

    size_t _n_feature;
    size_t _n_base_data;

    vector<uint8_t, IFlib::aligned_allocator<uint8_t, 64>> sparse_ik_flat; // 连续内存存储并设置起始地址为 32 字节对齐
    vector<uint8_t *> sparse_ik_data_ptr;                           // 行指针数组

public:
    Exhaustive_search_IForest_Processor(size_t psi = 1024, size_t t = 100) : _psi(psi), _t(t) {}
    Exhaustive_search_IForest_Processor(IFlib::SparseDataUINT8 &raw_base_data, size_t psi = 1024, size_t t = 100)
        : _psi(psi), _t(t)
    {
        // 转移数据所有权
        sparse_ik_flat = move(raw_base_data.data_flat);
        sparse_ik_data_ptr = move(raw_base_data.data_ptrs);

        // 设置维度参数
        _n_feature = raw_base_data.n_features;
        _n_base_data = raw_base_data.n_samples;
    }

    void init(IFlib::SparseDataUINT8 &raw_base_data)
    {
        // 转移数据所有权
        sparse_ik_flat = move(raw_base_data.data_flat);
        sparse_ik_data_ptr = move(raw_base_data.data_ptrs);

        // 设置维度参数
        _n_feature = raw_base_data.n_features;
        _n_base_data = raw_base_data.n_samples;
    }

    vector<vector<size_t>> query(const uint8_t *xq, size_t _n_query_data, size_t k = 100)
    {
        IFlib::BatchMinHeapHandler res_handler(_n_query_data, k);
        const size_t bs_x = 4096;
        const size_t bs_y = 1024;

        unique_ptr<int[]> ip_block(new int[bs_x * bs_y]);

        uint64_t total_gemc_ns = 0;

        for (size_t i0 = 0; i0 < _n_query_data; i0 += bs_x)
        {
            size_t i1 = min(_n_query_data, i0 + bs_x);
            res_handler.begin_batch(i0, i1);

            for (size_t j0 = 0; j0 < _n_base_data; j0 += bs_y)
            {
                size_t j1 = min(_n_base_data, j0 + bs_y);

                auto t0 = std::chrono::steady_clock::now();

                // 用 GEMC 计算比较结果，结果写入 ip_block
                {
                    size_t M = i1 - i0, N = j1 - j0;
                    gemc_simd_parallel_1x4_v4(xq + i0 * _n_feature,
                                              sparse_ik_data_ptr[j0],
                                              ip_block.get(),
                                              M, N, _n_feature);
                }

                // 打点结束
                auto t1 = std::chrono::steady_clock::now();
                total_gemc_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

                res_handler.add_results(j0, j1, ip_block.get());
            }
        }

        std::cout << "Total GEMC time: "
                  << (total_gemc_ns * 1e-9) << " s."
                  << endl;

        return res_handler.sort_all();
    }
};

int main(int argc, char *argv[])
{
    // 超参数定义
    size_t t = 4096;
    size_t psi = 4;
    std::string dataset_name = "fiqa_4096_qwen3";
    bool normlization = false;
    
    size_t m = 500;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        // 带选项的解析方式（例如：--t 2048 --psi 32 --dataset custom_name）
        if (arg == "--t" || arg == "-t")
        {
            if (++i < argc)
                t = std::strtoull(argv[i], nullptr, 10);
        }
        else if (arg == "--psi" || arg == "-p")
        {
            if (++i < argc)
                psi = std::strtoull(argv[i], nullptr, 10);
        }
        else if (arg == "--m" || arg == "-m")
        {
            if (++i < argc)
                m = std::strtoull(argv[i], nullptr, 10);
        }
        else if (arg == "--dataset" || arg == "-d")
        {
            if (++i < argc)
                dataset_name = argv[i];
        }
        else if (arg == "--norm" || arg == "-norm")
        {
            if (i + 1 < argc)
            {
                std::string val = argv[++i];
                if (val == "true" || val == "1")
                    normlization = true;
                else if (val == "false" || val == "0")
                    normlization = false;
                else
                {
                    std::cerr << "Invalid value for " << arg << ": " << val
                              << " (expected true/false or 1/0)\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }

    ifstream json_file("/home/yfwang/zzb/ann/MinibatchKmeans/dataset.json");
    if (!json_file.is_open())
    {
        throw std::runtime_error("无法打开 JSON 文件");
    }

    // 解析 JSON
    json json_data;
    json_file >> json_data;

    // 提取路径
    const string label_path = json_data[dataset_name]["uivecs_label_path"];
    const string base_data_path = json_data[dataset_name]["base_data_path"];
    const string query_data_path = json_data[dataset_name]["query_data_path"];
    const string dataset_type = json_data[dataset_name]["type"];

    size_t nb, nq, d;
    size_t k = 100;

    // 加载向量
    float *xb = read_fvecs(base_data_path, nb, d);
    float *xq = read_fvecs(query_data_path, nq, d);
    auto labels = read_uivecs(label_path);
    assert(m <= d);

    // ================================= iForest =================================
    IFlib::SparseDataUINT8 raw_base_data, query_raw_data;
    Exhaustive_search_IForest_Processor processor(psi, t);

    std::string base_transform_data_path =
        "/home/yfwang/zzb/ann/ann_dataset/IKE_data/VD_m/" +
        dataset_name + "_base_t_" + std::to_string(t) +
        "_psi_" + std::to_string(psi) + "_m_" + std::to_string(m) + ".bin";

    std::string query_transform_data_path =
        "/home/yfwang/zzb/ann/ann_dataset/IKE_data/VD_m/" +
        dataset_name + "_query_t_" + std::to_string(t) +
        "_psi_" + std::to_string(psi) + "_m_" + std::to_string(m) + ".bin";

    cout << "Begin iForest Mapping:" << '\n';
    auto map_start = chrono::high_resolution_clock::now();
    IFlib::VDMap_nobias model(xb, nb, d, m, t, psi);
    cout << "Model trained!" << '\n';
    raw_base_data = model.transform_with_cache(xb, nb, base_transform_data_path);
    processor.init(raw_base_data);
    auto map_end = chrono::high_resolution_clock::now();

    double index_time = chrono::duration<double>(map_end - map_start).count();
    cout << "Mapping took "
         << index_time // 直接转换为秒（浮点数）
         << " s" << endl;

    // ================================= Search =================================
    cout << "Begin Search:" << endl;
    auto search_start = chrono::high_resolution_clock::now();
    query_raw_data = model.transform_with_cache(xq, nq, query_transform_data_path);
    vector<vector<size_t>> query_res = processor.query(query_raw_data.data_ptrs[0],
                                                       query_raw_data.n_samples,
                                                       k);
    auto search_end = chrono::high_resolution_clock::now();

    double search_time = chrono::duration<double>(search_end - search_start).count();
    cout << "Search took "
         << search_time // 直接转换为秒（浮点数）
         << " s" << endl;

    // ================================= Metric =================================
    vector<vector<uint32_t>> scores;
    if (dataset_type != "binary")
    {
        const string score_path = json_data[dataset_name]["uivecs_score_path"];
        scores = read_uivecs(score_path);
    }

    size_t R_list[] = {1, 10, k};
    vector<double> recall_list;
    vector<double> mrr_list;
    vector<double> ndcg_list;

    double recall_sum = 0., mrr_sum = 0., ndcg_sum = 0.;

    for (size_t R : R_list)
    {
        double recall = compute_recall_at_r(labels, query_res, R);
        double mrr = compute_mrr(labels, query_res, R);
        double ndcg = compute_ndcg(labels, query_res, R, scores);

        recall_sum += recall;
        mrr_sum += mrr;
        ndcg_sum += ndcg;

        recall_list.push_back(recall);
        mrr_list.push_back(mrr);
        ndcg_list.push_back(ndcg);
    }

    delete[] xb;
    delete[] xq;

    // ================================= Write CSV =================================

    // 打开文件（追加模式）
    string csv_filename = "/home/yfwang/zzb/ann/psKC-C++/VD_res/" + dataset_name + ".csv";
    std::ofstream file(csv_filename, std::ios::app);

    if (!file.is_open())
    {
        throw std::runtime_error("无法打开文件: " + csv_filename);
    }

    // 写入CSV行
    file << t << ",";
    file << psi << ",";
    file << m << ",";
    file << std::fixed << std::setprecision(4);
    file << index_time << ",";
    file << search_time << ",";
    file << std::fixed << std::setprecision(2);

    for (auto item : recall_list)
        file << item * 100 << ",";
    file << recall_sum * 100 / recall_list.size() << ",";

    for (auto item : mrr_list)
        file << item * 100 << ",";
    file << mrr_sum * 100 / mrr_list.size() << ",";

    for (auto item : ndcg_list)
        file << item * 100 << ",";
    file << ndcg_sum * 100 / ndcg_list.size() << "\n";
}