/*
hnsw_search and hnsw_add_vertices functions are partly based on the FAISS project:
 https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.cpp
*/

#include "IForest_index/IForest_HNSW.h"
#include "tools/ResHandler.h"
#include <chrono>
#include <cassert>
#include <memory>


namespace IFlib{

    using storage_idx_t = HNSW::storage_idx_t;
    using NodeDistFarther = HNSW::NodeDistFarther;

    namespace
    {
        std::unique_ptr<IForestSimComputer> create_sim_computer(size_t psi, 
            size_t d, const uint8_t *x)
        {
            if (psi == 2)
                return std::make_unique<IForestSimComputerBit>(d, x);
            else if (psi <= 4)
                return std::make_unique<IForestSimComputer2Bit>(d, x);
            else if (psi <= 16)
                return std::make_unique<IForestSimComputer4Bit>(d, x);
            else
                return std::make_unique<IForestSimComputerUint8>(d, x);
        }

        void hnsw_add_vertices(
            IForestHNSW &index_hnsw,
            size_t n0,
            size_t n,
            const uint8_t *x,
            bool verbose,
            bool preset_levels = false)
        {
            size_t d = index_hnsw.d;
            HNSW &hnsw = index_hnsw.hnsw;
            size_t ntotal = n0 + n;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (verbose)
            {
                printf("hnsw_add_vertices: adding %zd elements on top of %zd "
                       "(preset_levels=%d)\n",
                       n,
                       n0,
                       int(preset_levels));
            }

            if (n == 0)
            {
                return;
            }

            int max_level = hnsw.prepare_level_tab(n, preset_levels);

            if (verbose)
            {
                printf("  max_level = %d\n", max_level);
            }

            std::vector<omp_lock_t> locks(ntotal);
            for (int i = 0; i < ntotal; i++)
                omp_init_lock(&locks[i]);

            // add vectors from highest to lowest level
            std::vector<int> hist;
            std::vector<int> order(n);

            { // make buckets with vectors of the same level

                // build histogram
                for (int i = 0; i < n; i++)
                {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    while (pt_level >= hist.size())
                        hist.push_back(0);
                    hist[pt_level]++;
                }

                // accumulate
                std::vector<int> offsets(hist.size() + 1, 0);
                for (int i = 0; i < hist.size() - 1; i++)
                {
                    offsets[i + 1] = offsets[i] + hist[i];
                }

                // bucket sort
                for (int i = 0; i < n; i++)
                {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    order[offsets[pt_level]++] = pt_id;
                }
            }

            // idx_t check_period = InterruptCallback::get_period_hint(
            //     max_level * index_hnsw.d * hnsw.efConstruction);

            { // perform add
                RandomGenerator rng2(789);

                int i1 = n;

                for (int pt_level = hist.size() - 1;
                     pt_level >= 0;
                     pt_level--)
                {
                    int i0 = i1 - hist[pt_level];

                    if (verbose)
                    {
                        printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
                    }

                    // random permutation to get rid of dataset order bias
                    for (int j = i0; j < i1; j++)
                        std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

                    bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
                {
                    VisitedTable vt(ntotal);
                    auto sim_computer = create_sim_computer(index_hnsw.psi, index_hnsw.d, x);
                    int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                    size_t counter = 0;

                    // here we should do schedule(dynamic) but this segfaults for
                    // some versions of LLVM. The performance impact should not be
                    // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                    for (int i = i0; i < i1; i++)
                    {
                        storage_idx_t pt_id = order[i];
                        sim_computer->set_query(x + (pt_id - n0) * d);

                        // cannot break
                        if (interrupt)
                        {
                            continue;
                        }

                        hnsw.add_with_locks(
                            *sim_computer,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0));

                        if (prev_display >= 0 && i - i0 > prev_display + 10000)
                        {
                            prev_display = i - i0;
                            printf("  %d / %d\r", i - i0, i1 - i0);
                            fflush(stdout);
                        }
                        // if (counter % check_period == 0)
                        // {
                        //     if (InterruptCallback::is_interrupted())
                        //     {
                        //         interrupt = true;
                        //     }
                        // }
                        counter++;
                    }
                }
                // if (interrupt)
                // {
                //     FAISS_THROW_MSG("computation interrupted");
                // }
                i1 = i0;
            }
        }
        if (verbose)
        {
            auto t1 = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(t1 - t0).count();
            printf("Done in %.3f s\n", duration);
        }

        for (int i = 0; i < ntotal; i++)
        {
            omp_destroy_lock(&locks[i]);
        }
    }

    void hnsw_search(
        const IForestHNSW *index,
        size_t n,
        const float *xq,
        size_t d_,
        BatchMinHeapHandler &bres)
    {
        const HNSW &hnsw = index->hnsw;

        int efSearch = hnsw.efSearch;

        size_t psi = index->psi;

        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        size_t check_period = (size_t)1 << 30;

        for (size_t i0 = 0; i0 < n; i0 += check_period)
        {
            size_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
            {
                VisitedTable vt(index->ntotal);
                BatchMinHeapHandler::SingleResultHandler res(bres);
                auto sim_computer = create_sim_computer(psi, index->d, index->base_data);
                std::vector<uint8_t> x;

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
                for (size_t i = i0; i < i1; i++)
                {
                    res.begin(i);
                    if (psi == 2){
                        x = index->model->transform_single_point_packed(xq + i * d_);
                    } 
                    else if (psi <= 4){
                        x = index->model->transform_single_point_packed_2bit(xq + i * d_);
                    }
                    else if (psi <= 16){
                        x = index->model->transform_single_point_packed_4bit(xq + i * d_);
                    }
                    else{
                        x = index->model->transform_single_point(xq + i * d_);
                    }
                    sim_computer->set_query(x.data());

                    HNSWStats stats = hnsw.search(*sim_computer, res, vt);
                    n1 += stats.n1;
                    n2 += stats.n2;
                    ndis += stats.ndis;
                    nhops += stats.nhops;
                }
            }
            // InterruptCallback::check();
        }

        // hnsw_stats.combine({n1, n2, ndis, nhops});
    }

}

    IForestHNSW::IForestHNSW(size_t t, size_t psi, int M, bool verbose)
        :t(t), psi(psi), verbose(verbose), hnsw(M) 
    {
        d = calculateCodeSize(t, psi);
    }

    void IForestHNSW::add(size_t n, const float *x)
    {
        size_t n0 = ntotal;
        ntotal = n0 + n;
        encoded_base_data.resize(ntotal * d); // d is code size
        base_data = encoded_base_data.data() + n0 * d;

        if (psi == 2)
        {
            model->transform_packed(n, x, base_data);
        }
        else if (psi <= 4)
        {
            model->transform_packed_2bit(n, x, base_data);
        }
        else if (psi <= 16)
        {
            model->transform_packed_4bit(n, x, base_data);
        }
        else
        {
            model->transform(n, x, base_data);
        }

        hnsw_add_vertices(*this, n0, n, base_data, verbose, hnsw.levels.size() == ntotal);
    }

    std::vector<std::vector<size_t>> IForestHNSW::search(
        size_t n,
        const float *xq,
        size_t d_,
        size_t k) const
    {

        BatchMinHeapHandler bres(n, k);

        hnsw_search(this, n, xq, d_, bres);

        return bres.sort_all();
    }

} // namespace IFlib