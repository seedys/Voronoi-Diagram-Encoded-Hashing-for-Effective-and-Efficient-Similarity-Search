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
#include "IForest_index/IForest_Flat.h"

#include "utils.h"


using json = nlohmann::json;
using std::vector, std::string, std::cout, std::endl;

int main(int argc, char *argv[])
{
    // Hyperparameter definition
    size_t t = 4096;
    size_t psi = 15;

    int repeat_times = 10; // Repeat the search process multiple times and calculate the average running time
    int warmup_times = 3;
    int seed = 0;

    string corpus_data_path, query_data_path, label_path, score_path;

    // command-line parameters
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

        else if (arg == "--repeat" || arg == "-r")
        {
            if (++i < argc)
                repeat_times = stoi(argv[i]);
        }
        else if (arg == "--warmup")
        {
            if (++i < argc)
                warmup_times = stoi(argv[i]);
        }
        else if (arg == "--seed" || arg == "-sd")
        {
            if (++i < argc)
                seed = stoi(argv[i]);
        }
    }

    // ================================= IKE =================================
    vector<vector<float>> base_data = read_fvecs(corpus_data_path);
    vector<vector<float>> query_data = read_fvecs(query_data_path);

    cout << "Begin indexing :" << endl;
    auto map_start = chrono::high_resolution_clock::now();
    IFlib::IForestFlat index(base_data, t, psi, false, seed);
    auto map_end = chrono::high_resolution_clock::now();

    double index_time = chrono::duration<double>(map_end - map_start).count();

    cout << "Indexing took " << index_time << " s\n";

    // ================================= Search =================================
    cout << "Begin Search:" << endl;
    size_t k = 100;
    vector<vector<size_t>> query_res;
    double total_time = 0.0;

    // warmup
    for (int i = 0; i < warmup_times; i++)
    {
        query_res = index.query(query_data, k);
    }
    
    for (int i = 0; i < repeat_times; i++)
    {
        auto search_start = chrono::high_resolution_clock::now();
        query_res = index.query(query_data, k);
        auto search_end = chrono::high_resolution_clock::now();

        total_time += chrono::duration<double>(search_end - search_start).count();
    }
    double search_time = total_time / repeat_times;
    cout << "Search time: " << search_time << " s\n";

    // ================================= Metric =================================
    vector<vector<uint32_t>> labels = read_uivecs(label_path);
    vector<vector<uint32_t>> scores;
    if (!score_path.empty())
    {
        scores = read_uivecs(score_path);
    }

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
        cout << "MRR@" << R_list[i] << ": "
             << std::fixed << std::setprecision(2) << mrr_list[i]
             << std::defaultfloat << '\n';

        cout << "nDCG@" << R_list[i] << ": "
             << std::fixed << std::setprecision(2) << ndcg_list[i]
             << std::defaultfloat << '\n';
    }
}