#include "IForest_index/IForest_HNSW.h"
#include <nlohmann/json.hpp>
#include <utils.h>

#include <thread>
#include <iostream>
#include <malloc.h>

using json = nlohmann::json;
using std::vector, std::string, std::cout, std::endl;

int main(int argc, char *argv[])
{
    // Hyperparameter definition
    size_t t = 4096;
    size_t psi = 2;
    int M = 32;
    int efConstruction = 500;
    int repeat_times = 10; // Repeat the search process multiple times and calculate the average running time
    int max_threads = omp_get_max_threads();
    int seed = 0;

    string corpus_data_path, query_data_path, label_path, score_path;

    vector<int> efSearch_list = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 30, 40, 50, 100, 200, 400, 800, 1600, 2000, 4000, 6000};

    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
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

        else if (arg == "--corpus_data_path" || arg == "-corpus")
        {
            if (++i < argc)
                corpus_data_path = argv[i];
        }
        else if (arg == "--query_data_path" || arg == "-query")
        {
            if (++i < argc)
                query_data_path = argv[i];
        }
        else if (arg == "--label_path" || arg == "-label")
        {
            if (++i < argc)
                label_path = argv[i];
        }
        else if (arg == "--score_path" || arg == "-score")
        {
            if (++i < argc)
                score_path = argv[i];
        }

        else if (arg == "--M" || arg == "-m")
        {
            if (++i < argc)
                M = std::stoi(argv[i]);
        }
        else if (arg == "--efConstruction" || arg == "-c")
        {
            if (++i < argc)
                efConstruction = std::stoi(argv[i]);
        }
        else if (arg == "--repeat" || arg == "-r")
        {
            if (++i < argc)
                repeat_times = stoi(argv[i]);
        }
        else if (arg == "--seed" || arg == "-sd")
        {
            if (++i < argc)
                seed = stoi(argv[i]);
        }
    }

    // ================================= index =================================
    size_t nq, d, nb;
    size_t k = 100;

    float *xb = read_fvecs(corpus_data_path, nb, d);
    float *xq = read_fvecs(query_data_path, nq, d);

    vector<const float *> data_ptr(nb);
    for (size_t i = 0; i < nb; i++)
        data_ptr[i] = xb + i * d;

    omp_set_num_threads(max_threads);

    cout << "Begin index HNSW (IKE) building with t=" << t << ", psi=" << psi << ", seed=" << seed << ":\n";
    auto build_start = chrono::high_resolution_clock::now();

    IFlib::IForestMap_nobias model(data_ptr, t, psi, d, false, seed);
    IFlib::IForestHNSW processor(t, psi, M, true);
    processor.model = &model;
    processor.hnsw.efConstruction = efConstruction;
    processor.add(nb, xb);

    auto build_end = chrono::high_resolution_clock::now();
    double index_time = chrono::duration<double>(build_end - build_start).count();

    cout << "Indexing took " << index_time << " s\n";

    // ================================= Search =================================
    vector<vector<uint32_t>> labels = read_uivecs(label_path);
    vector<vector<uint32_t>> scores;
    if (!score_path.empty())
    {
        scores = read_uivecs(score_path);
    }
    vector<vector<size_t>> query_res;

    for (auto tmp_efSearch : efSearch_list)
    {
        omp_set_num_threads(max_threads);
        processor.hnsw.efSearch = tmp_efSearch;

        query_res = processor.search(nq, xq, d, k);
        double total_time = 0.0;
        for (int i = 0; i < repeat_times; i++)
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            query_res = processor.search(nq, xq, d, k);
            auto t3 = std::chrono::high_resolution_clock::now();

            double tmp_search_time = std::chrono::duration<double>(t3 - t2).count();
            total_time += tmp_search_time;
        }
        double search_time = total_time / repeat_times;

        cout << std::fixed << std::setprecision(6);
        cout << "Search with efSearch = " << tmp_efSearch << " took "
                << search_time << " s";

        // ================================= Metric =================================
        vector<size_t> R_list = {1, 10, k};
        vector<double> mrr_list;
        vector<double> ndcg_list;

        for (size_t R : R_list)
        {
            double mrr = compute_mrr_at_r(labels, query_res, R);
            double ndcg = compute_ndcg_at_r(labels, query_res, R, scores);

            mrr_list.push_back(mrr * 100);
            ndcg_list.push_back(ndcg * 100);
        }

        for (size_t i = 0; i < R_list.size(); i++)
        {
            cout << ", MRR@" << R_list[i] << ": "
                 << std::fixed << std::setprecision(2) << mrr_list[i]
                 << std::defaultfloat;

            cout << ", nDCG@" << R_list[i] << ": "
                 << std::fixed << std::setprecision(2) << ndcg_list[i]
                 << std::defaultfloat;
        }

        cout << ";\n";
    }

    cout << "\n";

    delete[] xq;
    delete[] xb;

    return 0;
}