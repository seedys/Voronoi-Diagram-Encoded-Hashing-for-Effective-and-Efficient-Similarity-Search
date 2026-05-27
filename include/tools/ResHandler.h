#ifndef RES_HANDLER_H
#define RES_HANDLER_H

#include <cstddef>
#include <vector>
#include <queue>
#include <functional>
#include <utility>
#include <omp.h>
#include <cstring>

namespace IFlib{
    struct BatchMinHeapHandler{
        using Element = std::pair<int, std::size_t>;
        using MINPQ = std::priority_queue<
            Element,
            std::vector<Element>,
            std::greater<Element>>;

        std::size_t nq;
        std::size_t k;
        std::vector<MINPQ> heap_vector;
        std::size_t i0, i1;

        BatchMinHeapHandler(std::size_t nq_, std::size_t k_)
            : nq(nq_), k(k_), heap_vector(nq_)
        {
    #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < nq; ++i){
                std::vector<Element> buf;
                // buf.reserve(k);
                buf.resize(k, Element{-1, 0});
                heap_vector[i] = MINPQ(std::greater<Element>{}, std::move(buf));
            }
        }

        /******************************************************
         * API for 1 result at a time (each SingleResultHandler is
         * called from 1 thread)
         */

        struct SingleResultHandler
        {
            BatchMinHeapHandler &hr;
            size_t k;

            MINPQ *min_heap;
            int threshold;

            explicit SingleResultHandler(BatchMinHeapHandler &hr)
                : hr(hr), k(hr.k) {}

            /// begin results for query # i
            void begin(size_t i)
            {
                min_heap = &hr.heap_vector[i];
                threshold = min_heap->top().first;
            }

            /// add one result for query i
            void add_result(int sim, size_t idx)
            {
                if (sim > threshold)
                {
                    min_heap->pop();
                    min_heap->emplace(sim, idx);
                    threshold = min_heap->top().first;
                }
            }

        };

        // Set the row range for the next add_desults [i0, i1)
        void begin_batch(size_t i0_, size_t i1_){
            i0 = i0_;
            i1 = i1_;
        }

        // Processing the corpus data of [j0, j1)
        void add_results(size_t j0, size_t j1, const int *sim_tab)
        {
    #pragma omp parallel for
            for (size_t i = i0; i < i1; i++)
            {
                const int *sim_tab_i = sim_tab + (j1 - j0) * (i - i0) - j0;
                MINPQ &sim_heap = heap_vector[i];
                // int thresh = 0;
                for (size_t j = j0; j < j1; j++)
                {
                    int sim = sim_tab_i[j];

                    if (sim > sim_heap.top().first)
                    {
                        sim_heap.pop();
                        sim_heap.emplace(sim, j);
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> sort_all(){
            std::vector<std::vector<size_t>> res(nq);

    #pragma omp parallel for
            for (size_t i = 0; i < nq; ++i)
            {
                MINPQ &sim_heap = heap_vector[i];
                std::vector<size_t> top_k(k);
                for (int idx = k - 1; idx >= 0; --idx)
                {
                    top_k[idx] = sim_heap.top().second;
                    sim_heap.pop();
                }
                res[i] = std::move(top_k);
            }

            return res;
        }
    };

    struct BatchTop1Handler{
        int* sims;
        std::size_t *ids;

        std::size_t nq;
        std::size_t i0, i1;

        BatchTop1Handler(std::size_t nq_, int* sim_tab, std::size_t* ids_tab)
            : nq(nq_), sims(sim_tab), ids(ids_tab) {}

        void reset(){
            std::memset(sims, 0, nq * sizeof(int));
            std::memset(ids, 0, nq * sizeof(std::size_t));
        }

        void begin_batch(std::size_t i0_, std::size_t i1_)
        {
            i0 = i0_;
            i1 = i1_;
        }

        // Processing the corpus data of [j0, j1)
        void add_results(size_t j0, size_t j1, const int *sim_tab)
        {
#pragma omp parallel for schedule(static)
            for (size_t i = i0; i < i1; i++)
            {
                const int *sim_tab_i = sim_tab + (j1 - j0) * (i - i0) - j0;
                
                int &max_sim = this->sims[i];
                size_t &max_index = this->ids[i];

                for (size_t j = j0; j < j1; j++)
                {
                    const int sim = sim_tab_i[j];

                    if (sim > max_sim)
                    {
                        max_sim = sim;
                        max_index = j;
                    }
                }
            }
        }
    };
} // namespace IFlib

#endif // RES_HANDLER_H