#include "IForest_index/IVFIForest.h"
#include "utils.h"
#include "nlohmann/json.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <iostream>
#include <thread>

using json = nlohmann::json;
using std::vector, std::string, std::cout, std::endl;

int main(int argc, char *argv[])
{
    // Hyperparameter definition
    size_t t = 4096;
    size_t psi = 2;
    int nlist = 4096; // number of clusters
    int repeat_times = 10; // Repeat the search process multiple times and calculate the average running time
    int max_threads = omp_get_max_threads();
    int seed = 0;

    string corpus_data_path, query_data_path, label_path, score_path;

    vector<int> nprobe_list = {1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

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

        else if (arg == "--nlist")
        {
            if (++i < argc)
                nlist = stoi(argv[i]);
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


    size_t nb, nq, d;
    size_t k = 100;

    // load raw data
    float *xb = read_fvecs(corpus_data_path, nb, d);
    float *xq = read_fvecs(query_data_path, nq, d);

    vector<long> I(nq * k);
    vector<float> D(nq * k);

    // ======================= index =======================
    omp_set_num_threads(max_threads);

    cout << "Begin index IVF (IKE) building with t=" << t << ", psi=" << psi << ", seed=" << seed << ":\n";
    auto t0 = chrono::high_resolution_clock::now();
    // 1)  Coarse Quantizer
    faiss::IndexFlatL2 quantizer(d);

    // 2) build index
    IFlib::IVFIForest index(&quantizer, d, nlist, t, psi, false, seed);
    index.verbose = true;

    // train and add
    index.train(nb, xb);
    index.add(nb, xb);
    auto t1 = chrono::high_resolution_clock::now();
    double index_time = chrono::duration<double>(t1 - t0).count();

    cout << "Indexing took " << index_time << " s\n";

    // ======================= search =======================
    auto labels = read_uivecs(label_path);
    vector<vector<uint32_t>> scores;
    if (!scores.empty())
    {
        scores = read_uivecs(score_path);
    }
    vector<vector<size_t>> query_res(nq, vector<size_t>(k));
    for (auto tmp_nprobe : nprobe_list)
    {
        omp_set_num_threads(max_threads);
        index.nprobe = tmp_nprobe;

        index.search(nq, xq, k, D.data(), I.data());
        double total_time = 0.0;
        for (int i = 0; i < repeat_times; i++)
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            index.search(nq, xq, k, D.data(), I.data());
            auto t3 = std::chrono::high_resolution_clock::now();

            double tmp_search_time = std::chrono::duration<double>(t3 - t2).count();
            total_time += tmp_search_time;
        }
        double search_time = total_time / repeat_times;

        cout << std::fixed << std::setprecision(6);
        cout << "Search with nprobe = " << tmp_nprobe << " took " << search_time << " s";

        // ======================= res =======================
        for (size_t qi = 0; qi < nq; ++qi)
        {
            for (size_t j = 0; j < k; ++j)
            {
                query_res[qi][j] = static_cast<size_t>(I[qi * k + j]);
            }
        }

        // ======================= metric =======================
        vector<size_t> R_list = {1, 10, k};
        vector<double> mrr_list, ndcg_list;

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

    delete[] xb;
    delete[] xq;

    return 0;
}