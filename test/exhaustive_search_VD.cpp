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
#include "IK/VD.cpp"
#include "utils.h"

using json = nlohmann::json;
using namespace std;

class Exhaustive_search_VD_Processor
{
private:
    int _psi;
    int _t;

    size_t _n_feature;
    size_t _n_base_data;
    size_t _n_query_data;
    vector<uint8_t, IFlib::aligned_allocator<uint8_t, 64>> sparse_ik_flat;    // 连续内存存储并设置起始地址为 64 字节对齐
    vector<uint8_t, IFlib::aligned_allocator<uint8_t, 64>> sparse_query_flat; 

public:
    Exhaustive_search_VD_Processor(size_t psi = 16, size_t t = 4096) : _psi(psi), _t(t) {}
    Exhaustive_search_VD_Processor(IFlib::SparseDataUINT8 &raw_base_data, 
        IFlib::SparseDataUINT8 &raw_query_data, 
        size_t psi = 16, 
        size_t t = 4096)
        : _psi(psi), _t(t)
    {
        // 转移数据所有权
        sparse_ik_flat = move(raw_base_data.data_flat);

        sparse_query_flat = move(raw_query_data.data_flat);

        // 设置维度参数
        _n_feature = raw_base_data.n_features;
        _n_base_data = raw_base_data.n_samples;

        _n_query_data = raw_query_data.n_samples;
    }

    void init(IFlib::SparseDataUINT8 &raw_base_data, IFlib::SparseDataUINT8 &raw_query_data)
    {
        // 转移数据所有权
        sparse_ik_flat = move(raw_base_data.data_flat);

        sparse_query_flat = move(raw_query_data.data_flat);

        // 设置维度参数
        _n_feature = raw_base_data.n_features;
        _n_base_data = raw_base_data.n_samples;

        _n_query_data = raw_query_data.n_samples;
    }

    vector<vector<size_t>> query(size_t k = 100)
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
                    gemc_simd_parallel_1x4_v4(sparse_query_flat.data() + i0 * _n_feature,
                                              sparse_ik_flat.data() + j0 * _n_feature,
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
    size_t psi = 256;
    std::string dataset_name = "feverHN_4096_sup";

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
        else if (arg == "--dataset" || arg == "-d")
        {
            if (++i < argc)
                dataset_name = argv[i];
        }
    }

    ifstream json_file("/home/yfwang/zzb/ann/psKC-C++/dataset.json");
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

    // ================================= VD =================================
    IFlib::SparseDataUINT8 raw_base_data, query_raw_data;
    Exhaustive_search_VD_Processor processor(psi, t);
    float *xb;
    float *xq;
    double index_time = 0.0;
    {
        size_t nb, nq, d;
        xb = read_fvecs(base_data_path, nb, d);
        xq = read_fvecs(query_data_path, nq, d);

        cout << "Begin iForest Mapping:" << endl;
        auto map_start = chrono::high_resolution_clock::now();
        IFlib::VDMap_nobias model(xb, nb, d, t, psi);

        std::string base_transform_data_path =
            "/data3/yfwang/zzb/ann/ann_dataset/IKE_data/VD/" +
            dataset_name + "_base_t_" + std::to_string(t) +
            "_psi_" + std::to_string(psi) + ".bin";

        std::cout << "Begin base Transform:\n";
        raw_base_data = model.transform_with_cache(xb, nb, base_transform_data_path);

        std::string query_transform_data_path =
            "/data3/yfwang/zzb/ann/ann_dataset/IKE_data/VD/" +
            dataset_name + "_query_t_" + std::to_string(t) +
            "_psi_" + std::to_string(psi) + ".bin";

        std::cout << "Begin query Transform:\n";
        query_raw_data = model.transform_with_cache(xq, nq, query_transform_data_path);

        processor.init(raw_base_data, query_raw_data);
        auto map_end = chrono::high_resolution_clock::now();
        index_time = chrono::duration<double>(map_end - map_start).count();

        cout << "Mapping took "
             << index_time  // 直接转换为秒（浮点数）
             << " s" << endl;
    }
    delete[] xb; // xb 是 new float[] 分配的
    delete[] xq;

    // ================================= Search =================================
    cout << "Begin Search:" << endl;
    size_t k = 100;
    auto search_start = chrono::high_resolution_clock::now();
    vector<vector<size_t>> query_res = processor.query(k);
    auto search_end = chrono::high_resolution_clock::now();

    double search_time = chrono::duration<double>(search_end - search_start).count();
    cout << "Search took "
         << search_time
         << " s" << endl;

    // ================================= Metric =================================
    vector<vector<uint32_t>> labels = read_uivecs(label_path);
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

    // cout << fixed << setprecision(4); // 全局设置，后续输出均生效
    // for (size_t i = 0; i < recall_list.size(); ++i)
    //     cout << "Recall@" << R_list[i] << " : " << recall_list[i] << endl;

    // cout << "Avg. Recall : " << recall_sum / recall_list.size() << endl;

    // cout << "====================" << endl;
    // for (size_t i = 0; i < mrr_list.size(); ++i)
    //     cout << "MRR@" << R_list[i] << " : " << mrr_list[i] << endl;
    // cout << "Avg. MRR : " << mrr_sum / mrr_list.size() << endl;

    // cout << "====================" << endl;
    // for (size_t i = 0; i < ndcg_list.size(); ++i)
    //     cout << "nDCG@" << R_list[i] << " : " << ndcg_list[i] << endl;
    // cout << "Avg. nDCG : " << ndcg_sum / ndcg_list.size() << endl;

    // cout.unsetf(ios::fixed);
    // cout << setprecision(6); // 恢复默认精度

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
