/*
This implementation is based on the FAISS project: https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp
*/

#include "IForest_index/myHNSW.h"
#include <cassert>

namespace IFlib
{
    using storage_idx_t = HNSW::storage_idx_t;
    /**************************************************************
     * Other function
     **************************************************************/

    namespace
    {
        using storage_idx_t = HNSW::storage_idx_t;
        using NodeDistCloser = HNSW::NodeDistCloser;
        using NodeDistFarther = HNSW::NodeDistFarther;

        /// remove neighbors from the list to make it smaller than max_size
        void shrink_neighbor_list_out(
            IForestSimComputer &qdis,
            std::priority_queue<NodeDistFarther> &resultSet1,
            int max_size,
            bool keep_max_size_level0 = false)
        {
            if (resultSet1.size() < max_size)
            {
                return;
            }
            std::priority_queue<NodeDistCloser> resultSet;
            std::vector<NodeDistCloser> returnlist;

            while (resultSet1.size() > 0)
            {
                resultSet.emplace(resultSet1.top().sim, resultSet1.top().id);
                resultSet1.pop();
            }

            HNSW::shrink_neighbor_list(
                qdis, resultSet, returnlist, max_size, keep_max_size_level0);

            for (NodeDistCloser curen2 : returnlist)
            {
                resultSet1.emplace(curen2.sim, curen2.id);
            }
        }

        /// add a link between two elements, possibly shrinking the list
        /// of links to make room for it.
        void add_link(
            HNSW &hnsw,
            IForestSimComputer &qdis,
            storage_idx_t src,
            storage_idx_t dest,
            int level,
            bool keep_max_size_level0 = false)
        {
            size_t begin, end;
            hnsw.neighbor_range(src, level, &begin, &end);
            if (hnsw.neighbors[end - 1] == -1)
            {
                // there is enough room, find a slot to add it
                size_t i = end;
                while (i > begin)
                {
                    if (hnsw.neighbors[i - 1] != -1)
                        break;
                    i--;
                }
                hnsw.neighbors[i] = dest;
                return;
            }

            // otherwise we let them fight out which to keep

            // copy to resultSet...
            std::priority_queue<NodeDistFarther> resultSet;
            resultSet.emplace(qdis.symmetric_sim(src, dest), dest);
            for (size_t i = begin; i < end; i++)
            { // HERE WAS THE BUG
                storage_idx_t neigh = hnsw.neighbors[i];
                resultSet.emplace(qdis.symmetric_sim(src, neigh), neigh);
            }

            shrink_neighbor_list_out(qdis, resultSet, end - begin, keep_max_size_level0);

            // ...and back
            size_t i = begin;
            while (resultSet.size())
            {
                hnsw.neighbors[i++] = resultSet.top().id;
                resultSet.pop();
            }
            // they may have shrunk more than just by 1 element
            while (i < end)
            {
                hnsw.neighbors[i++] = -1;
            }
        }

    } // namespace

    /**************************************************************
     * HNSW structure implementation
     **************************************************************/
    int HNSW::nb_neighbors(int layer_no) const
    {
        if (layer_no < 0 || layer_no + 1 >= cum_nneighbor_per_level.size())
        {
            throw std::out_of_range("HNSW::nb_neighbors: invalid layer_no");
        }

        return cum_nneighbor_per_level[layer_no + 1] -
               cum_nneighbor_per_level[layer_no];
    }

    void HNSW::set_nb_neighbors(int level_no, int n)
    {
        if (levels.size() != 0)
        {
            throw std::logic_error("HNSW::set_nb_neighbors: levels must be empty");
        }

        int cur_n = nb_neighbors(level_no);
        for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++)
        {
            cum_nneighbor_per_level[i] += n - cur_n;
        }
    }

    int HNSW::cum_nb_neighbors(int layer_no) const
    {
        return cum_nneighbor_per_level[layer_no];
    }

    void HNSW::neighbor_range(size_t no, int layer_no, size_t *begin, size_t *end)
        const
    {
        size_t o = offsets[no];
        *begin = o + cum_nb_neighbors(layer_no);
        *end = o + cum_nb_neighbors(layer_no + 1);
    }

    HNSW::HNSW(int M) : rng(12345)
    {
        set_default_probas(M, 1.0 / log(M));
        offsets.push_back(0);
    }

    int HNSW::random_level()
    {
        double f = rng.rand_float();
        // could be a bit faster with bissection
        for (int level = 0; level < assign_probas.size(); level++)
        {
            if (f < assign_probas[level])
            {
                return level;
            }
            f -= assign_probas[level];
        }
        // happens with exponentially low probability
        return assign_probas.size() - 1;
    }

    void HNSW::set_default_probas(int M, float levelMult)
    {
        int nn = 0;
        cum_nneighbor_per_level.push_back(0);
        for (int level = 0;; level++)
        {
            float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            if (proba < 1e-9)
                break;
            assign_probas.push_back(proba);
            nn += level == 0 ? M * 2 : M;
            cum_nneighbor_per_level.push_back(nn);
        }
    }

    void HNSW::clear_neighbor_tables(int level)
    {
        for (int i = 0; i < levels.size(); i++)
        {
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            for (size_t j = begin; j < end; j++)
            {
                neighbors[j] = -1;
            }
        }
    }

    void HNSW::reset()
    {
        max_level = -1;
        entry_point = -1;
        offsets.clear();
        offsets.push_back(0);
        levels.clear();
        neighbors.clear();
    }

    int HNSW::prepare_level_tab(size_t n, bool preset_levels)
    {
        size_t n0 = offsets.size() - 1;

        if (preset_levels)
        {
            assert(n0 + n == levels.size());
        }
        else
        {
            assert(n0 == levels.size());
            for (int i = 0; i < n; i++)
            {
                int pt_level = random_level();
                levels.push_back(pt_level + 1);
            }
        }

        int max_level_2 = 0;
        for (int i = 0; i < n; i++)
        {
            int pt_level = levels[i + n0] - 1;
            if (pt_level > max_level_2)
                max_level_2 = pt_level;
            offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        }
        neighbors.resize(offsets.back(), -1);

        return max_level_2;
    }

    void HNSW::shrink_neighbor_list(
        IForestSimComputer &qdis,
        std::priority_queue<NodeDistCloser> &input,
        std::vector<NodeDistCloser> &output,
        int max_size,
        bool keep_max_size_level0)
    {
        std::vector<NodeDistCloser> outsiders;

        while (input.size() > 0)
        {
            NodeDistCloser v1 = input.top();
            input.pop();
            int sim_v1_q = v1.sim;

            bool good = true;
            for (NodeDistCloser v2 : output)
            {
                int sim_v1_v2 = qdis.symmetric_sim(v2.id, v1.id);

                if (sim_v1_v2 > sim_v1_q)
                {
                    good = false;
                    break;
                }
            }

            if (good)
            {
                output.push_back(v1);
                if (output.size() >= max_size)
                {
                    return;
                }
            }
            else if (keep_max_size_level0)
            {
                outsiders.push_back(v1);
            }
        }

        size_t idx = 0;

        while (keep_max_size_level0 && (output.size() < max_size) &&
               (idx < outsiders.size()))
        {
            output.push_back(outsiders[idx++]);
        }
    }

    /**************************************************************
     * Building, parallel
     **************************************************************/

    void HNSW::add_with_locks(
        IForestSimComputer &ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t> &locks,
        VisitedTable &vt,
        bool keep_max_size_level0)
    {
        //  greedy search on upper levels

        storage_idx_t nearest;
#pragma omp critical
        {
            nearest = entry_point;

            if (nearest == -1)
            {
                max_level = pt_level;
                entry_point = pt_id;
            }
        }

        if (nearest < 0)
        {
            return;
        }

        omp_set_lock(&locks[pt_id]);

        int level = max_level; // level at which we start adding neighbors
        int d_nearest = ptdis(nearest);

        for (; level > pt_level; level--)
        {
            greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        }

        for (; level >= 0; level--)
        {
            add_links_starting_from(
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks.data(),
                vt,
                keep_max_size_level0);
        }

        omp_unset_lock(&locks[pt_id]);

        if (pt_level > max_level)
        {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    void HNSW::add_links_starting_from(
        IForestSimComputer &ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        int d_nearest,
        int level,
        omp_lock_t *locks,
        VisitedTable &vt,
        bool keep_max_size_level0)
    {
        std::priority_queue<NodeDistFarther> link_targets;

        search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt);

        // but we can afford only this many neighbors
        int M = nb_neighbors(level);

        shrink_neighbor_list_out(ptdis, link_targets, M, keep_max_size_level0);

        std::vector<storage_idx_t> neighbors_to_add;
        neighbors_to_add.reserve(link_targets.size());
        while (!link_targets.empty())
        {
            storage_idx_t other_id = link_targets.top().id;
            add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
            neighbors_to_add.push_back(other_id);
            link_targets.pop();
        }

        omp_unset_lock(&locks[pt_id]);
        for (storage_idx_t other_id : neighbors_to_add)
        {
            omp_set_lock(&locks[other_id]);
            add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
            omp_unset_lock(&locks[other_id]);
        }
        omp_set_lock(&locks[pt_id]);
    }

    /**************************************************************
     * Searching
     **************************************************************/
    HNSWStats HNSW::search(
        IForestSimComputer &qdis,
        BatchMinHeapHandler::SingleResultHandler &res,
        VisitedTable &vt) const
    {
        HNSWStats stats;
        if (entry_point == -1)
        {
            return stats;
        }
        int k = res.k;

        bool bounded_queue = this->search_bounded_queue;
        int efSearch = this->efSearch;

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        int d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--)
        {
            HNSWStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            stats.combine(local_stats);
        }

        int ef = std::max(efSearch, k);
        if (bounded_queue)
        { // this is the most common branch
            MinimaxHeap candidates(ef);

            candidates.push(nearest, -d_nearest);

            search_from_candidates(
                *this, qdis, res, candidates, vt, stats, 0);
        }

        vt.advance();

        return stats;
    }

    using storage_idx_t = HNSW::storage_idx_t;
    using NodeDistCloser = HNSW::NodeDistCloser;
    using NodeDistFarther = HNSW::NodeDistFarther;
    HNSWStats greedy_update_nearest(
        const HNSW &hnsw,
        IForestSimComputer &qdis,
        int level,
        storage_idx_t &nearest,
        int &d_nearest)
    {
        HNSWStats stats;

        for (;;)
        {
            storage_idx_t prev_nearest = nearest;

            size_t begin, end;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            size_t ndis = 0;

            // a faster version: reference version in unit test test_hnsw.cpp
            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const int sim)
            {
                if (sim > d_nearest)
                {
                    nearest = idx;
                    d_nearest = sim;
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++)
            {
                storage_idx_t v = hnsw.neighbors[j];
                if (v < 0)
                    break;
                ndis += 1;

                buffered_ids[n_buffered] = v;
                n_buffered += 1;

                if (n_buffered == 4)
                {
                    int sim[4];
                    qdis.sim_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        sim[0],
                        sim[1],
                        sim[2],
                        sim[3]);

                    for (size_t id4 = 0; id4 < 4; id4++)
                    {
                        update_with_candidate(buffered_ids[id4], sim[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (size_t icnt = 0; icnt < n_buffered; icnt++)
            {
                int sim = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], sim);
            }

            // update stats
            stats.ndis += ndis;
            stats.nhops += 1;

            if (nearest == prev_nearest)
            {
                return stats;
            }
        }
    }

    void search_neighbors_to_add(
        HNSW &hnsw,
        IForestSimComputer &qdis,
        std::priority_queue<NodeDistFarther> &results,
        int entry_point,
        int d_entry_point,
        int level,
        VisitedTable &vt)
    {
        // top is nearest candidate
        std::priority_queue<NodeDistCloser> candidates;

        NodeDistCloser ev(d_entry_point, entry_point);
        candidates.push(ev);
        results.emplace(d_entry_point, entry_point);
        vt.set(entry_point);

        while (!candidates.empty())
        {
            // get nearest
            const NodeDistCloser &currEv = candidates.top();

            if (currEv.sim < results.top().sim)
            {
                break;
            }
            int currNode = currEv.id;
            candidates.pop();

            // loop over neighbors
            size_t begin, end;
            hnsw.neighbor_range(currNode, level, &begin, &end);

            // a faster version
            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const int sim)
            {
                if (results.size() < hnsw.efConstruction ||
                    results.top().sim < sim)
                {
                    results.emplace(sim, idx);
                    candidates.emplace(sim, idx);
                    if (results.size() > hnsw.efConstruction)
                    {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++)
            {
                storage_idx_t nodeId = hnsw.neighbors[j];
                if (nodeId < 0)
                    break;
                if (vt.get(nodeId))
                {
                    continue;
                }
                vt.set(nodeId);

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4)
                {
                    int sim[4];
                    qdis.sim_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        sim[0],
                        sim[1],
                        sim[2],
                        sim[3]);

                    for (size_t id4 = 0; id4 < 4; id4++)
                    {
                        update_with_candidate(buffered_ids[id4], sim[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (size_t icnt = 0; icnt < n_buffered; icnt++)
            {
                int sim = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], sim);
            }
        }

        vt.advance();
    }

    void search_from_candidates(
        const HNSW &hnsw,
        IForestSimComputer &qdis,
        BatchMinHeapHandler::SingleResultHandler &res,
        HNSW::MinimaxHeap &candidates,
        VisitedTable &vt,
        HNSWStats &stats,
        int level)
    {
        int ndis = 0;

        // can be overridden by search params
        bool do_dis_check = hnsw.check_relative_distance;
        int efSearch = hnsw.efSearch;

        int threshold = res.threshold;
        for (int i = 0; i < candidates.size(); i++)
        {
            size_t v1 = candidates.ids[i];
            int sim_v = -candidates.sim[i];
            assert(v1 >= 0);

            if(sim_v > threshold){
                res.add_result(sim_v, v1);
                threshold = res.threshold;
            }

            vt.set(v1);
        }

        int nstep = 0;
        size_t npopcc = 0;

        while (candidates.size() > 0)
        {
            int sim0 = 0;
            storage_idx_t v0 = candidates.pop_min(&sim0);
            npopcc++;

            if (do_dis_check)
            {
                // tricky stopping condition: there are more that ef
                // distances that are processed already that are smaller
                // than d0

                int n_dis_below = candidates.count_below(sim0);
                if (n_dis_below >= efSearch)
                {
                    // printf("%d \n", &npopcc);
                    break;
                }
            }

            size_t begin, end;
            hnsw.neighbor_range(v0, level, &begin, &end);

            // a faster version: reference version in unit test test_hnsw.cpp
            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t j = begin; j < end; j++)
            {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0)
                    break;

                __builtin_prefetch(vt.visited.data() + v1, 0, 2);
                jmax += 1;
            }

            int counter = 0;
            storage_idx_t saved_j[4];

            auto add_to_heap = [&](const size_t idx, const int sim)
            {
                if (sim > threshold)
                {
                    res.add_result(sim, idx);
                    threshold = res.threshold;
                }
                candidates.push(idx, -sim);
            };

            for (size_t j = begin; j < jmax; j++)
            {
                storage_idx_t v1 = hnsw.neighbors[j];

                bool vget = vt.get(v1);
                vt.set(v1);
                saved_j[counter] = v1;
                counter += vget ? 0 : 1;

                if (counter == 4)
                {
                    int sim[4];
                    qdis.sim_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        sim[0],
                        sim[1],
                        sim[2],
                        sim[3]);

                    for (size_t id4 = 0; id4 < 4; id4++)
                    {
                        add_to_heap(saved_j[id4], sim[id4]);
                    }

                    ndis += 4;
                    counter = 0;
                }
            }

            for (size_t icnt = 0; icnt < counter; icnt++)
            {
                int sim = qdis(saved_j[icnt]);
                add_to_heap(saved_j[icnt], sim);

                ndis += 1;
            }

            nstep++;
            if (!do_dis_check && nstep > efSearch)
            {
                break;
            }
        }

        if (level == 0)
        {
            stats.n1++;
            if (candidates.size() == 0)
            {
                stats.n2++;
            }
            stats.ndis += ndis;
            stats.nhops += nstep;
        }
    }


    /**************************************************************
     * MinimaxHeap
     **************************************************************/
namespace{
    bool cmp2(int a1, int b1, storage_idx_t a2, storage_idx_t b2)
    {
        return (a1 > b1) || ((a1 == b1) && (a2 > b2));
    }

    void heap_pop(size_t k, int *bh_val, storage_idx_t *bh_ids)
    {
        bh_val--; /* Use 1-based indexing for easier node->child translation */
        bh_ids--;
        int val = bh_val[k];
        storage_idx_t id = bh_ids[k];
        size_t i = 1, i1, i2;
        while (1)
        {
            i1 = i << 1;
            i2 = i1 + 1;
            if (i1 > k)
                break;
            if ((i2 == k + 1) ||
                cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2]))
            {
                if (cmp2(val, bh_val[i1], id, bh_ids[i1]))
                {
                    break;
                }
                bh_val[i] = bh_val[i1];
                bh_ids[i] = bh_ids[i1];
                i = i1;
            }
            else
            {
                if (cmp2(val, bh_val[i2], id, bh_ids[i2]))
                {
                    break;
                }
                bh_val[i] = bh_val[i2];
                bh_ids[i] = bh_ids[i2];
                i = i2;
            }
        }
        bh_val[i] = bh_val[k];
        bh_ids[i] = bh_ids[k];
    }

    /** Pushes the element (val, ids) into the heap bh_val[0..k-2] and
     * bh_ids[0..k-2].  on output the element at k-1 is defined.
     */
    void heap_push(
        size_t k,
        int *bh_val,
        storage_idx_t *bh_ids,
        int val,
        storage_idx_t id)
    {
        bh_val--; /* Use 1-based indexing for easier node->child translation */
        bh_ids--;
        size_t i = k, i_father;
        while (i > 1)
        {
            i_father = i >> 1;
            if (!cmp2(val, bh_val[i_father], id, bh_ids[i_father]))
            {
                /* the heap structure is ok */
                break;
            }
            bh_val[i] = bh_val[i_father];
            bh_ids[i] = bh_ids[i_father];
            i = i_father;
        }
        bh_val[i] = val;
        bh_ids[i] = id;
    }
} // namespace

    void HNSW::MinimaxHeap::push(storage_idx_t i, int v)
    {
        if (k == n)
        {
            if (v >= sim[0])
                return;
            if (ids[0] != -1)
            {
                --nvalid;
            }
            heap_pop(k--, sim.data(), ids.data());
        }
        heap_push(++k, sim.data(), ids.data(), v, i);
        ++nvalid;
    }

    float HNSW::MinimaxHeap::max() const
    {
        return sim[0];
    }

    int HNSW::MinimaxHeap::size() const
    {
        return nvalid;
    }

    void HNSW::MinimaxHeap::clear()
    {
        nvalid = k = 0;
    }

#ifdef USE_256_REG
    int HNSW::MinimaxHeap::pop_min(int *vmin_out)
    {
        assert(k > 0);

        int32_t min_dis = std::numeric_limits<int>::max();
        int32_t min_idx = -1;
        size_t iii = 0;

        // AVX2 uses 256-bit registers, processing 8 integers at once
        __m256i min_indices = _mm256_set1_epi32(-1);
        __m256i min_distances = _mm256_set1_epi32(std::numeric_limits<int>::max());
        __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i offset = _mm256_set1_epi32(8);

        // Main SIMD loop - processes 8 elements per iteration
        const int k8 = (k / 8) * 8;
        for (; iii < k8; iii += 8)
        {
            // Load 8 indices and distances
            __m256i indices = _mm256_loadu_si256((const __m256i *)(ids.data() + iii));
            __m256i distances = _mm256_loadu_si256((const __m256i *)(sim.data() + iii));

            // Create mask: mark invalid indices (index equals -1)
            __m256i invalid_mask = _mm256_cmpeq_epi32(indices, _mm256_set1_epi32(-1));

            // Create mask: mark distances greater than current minimum
            __m256i distance_mask = _mm256_cmpgt_epi32(distances, min_distances);

            // Combine masks: invalid indices OR distances greater than current minimum
            __m256i final_mask = _mm256_or_si256(invalid_mask, distance_mask);

            // Blend values using mask
            min_indices = _mm256_blendv_epi8(current_indices, min_indices, final_mask);
            min_distances = _mm256_blendv_epi8(distances, min_distances, final_mask);

            current_indices = _mm256_add_epi32(current_indices, offset);
        }

        // Extract values from AVX2 registers to arrays for processing
        int32_t min_indices_arr[8];
        int32_t min_distances_arr[8];
        _mm256_storeu_si256((__m256i *)min_indices_arr, min_indices);
        _mm256_storeu_si256((__m256i *)min_distances_arr, min_distances);

        // Find minimum value in extracted arrays
        for (size_t i = 0; i < 8; i++)
        {
            if (min_distances_arr[i] < min_dis ||
                (min_dis == min_distances_arr[i] && min_idx < min_indices_arr[i]))
            {
                min_dis = min_distances_arr[i];
                min_idx = min_indices_arr[i];
            }
        }

        // Process remaining elements (scalar approach)
        for (; iii < k; iii++)
        {
            if (ids[iii] == -1 && sim[iii] < min_dis)
            {
                min_dis = sim[iii];
                min_idx = iii;
            }
        }

        if (min_idx == -1)
            return -1;

        if (vmin_out)
        {
            *vmin_out = min_dis;
        }

        int ret = ids[min_idx];
        ids[min_idx] = -1;
        --nvalid;
        return ret;
    }
#else
    int HNSW::MinimaxHeap::pop_min(int *vmin_out)
    {
        assert(k > 0);

        int32_t min_idx = -1;
        int min_dis = std::numeric_limits<int>::max();

        __m512i min_indices = _mm512_set1_epi32(-1);
        __m512i min_distances = _mm512_set1_epi32(std::numeric_limits<int>::max());
        __m512i current_indices = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m512i offset = _mm512_set1_epi32(16);

        // Main SIMD loop
        const int k16 = (k / 16) * 16;
        for (size_t iii = 0; iii < k16; iii += 16)
        {
            __m512i indices = _mm512_loadu_si512((const __m512i *)(ids.data() + iii));
            __m512i distances = _mm512_loadu_epi32(sim.data() + iii);

            __mmask16 m1mask = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);
            __mmask16 dmask = _mm512_cmplt_epi32_mask(min_distances, distances); 

            __mmask16 finalmask = _mm512_kor(m1mask, dmask);

            min_indices = _mm512_mask_blend_epi32(finalmask, current_indices, min_indices);
            min_distances = _mm512_mask_blend_epi32(finalmask, distances, min_distances);

            current_indices = _mm512_add_epi32(current_indices, offset);
        }

        // Handle remaining elements
        if (k16 != k)
        {
            const __mmask16 kmask = (1 << (k - k16)) - 1;

            __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, ids.data() + k16);
            __m512i distances = _mm512_maskz_loadu_epi32(kmask, sim.data() + k16);

            __mmask16 m1mask = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);
            __mmask16 dmask = _mm512_cmplt_epi32_mask(min_distances, distances);

            __mmask16 finalmask = _mm512_kor(m1mask, dmask);

            min_indices = _mm512_mask_blend_epi32(finalmask, current_indices, min_indices);
            min_distances = _mm512_mask_blend_epi32(finalmask, distances, min_distances);
        }

        // Find minimum integer distance
        min_dis = _mm512_reduce_min_epi32(min_distances);

        // Find rightmost index with min distance
        __mmask16 mindmask = _mm512_cmpeq_epi32_mask(min_distances, _mm512_set1_epi32(min_dis));
        min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

        if (min_idx == -1)
            return -1;

        if (vmin_out)
        {
            *vmin_out = min_dis; // use int type output
        }

        int ret = ids[min_idx];
        ids[min_idx] = -1;
        --nvalid;
        return ret;
    }

    // int HNSW::MinimaxHeap::pop_min(int *vmin_out)
    // {
    //     assert(k > 0);
    //     // returns min. This is an O(n) operation
    //     int i = k - 1;
    //     while (i >= 0)
    //     {
    //         if (ids[i] != -1)
    //         {
    //             break;
    //         }
    //         i--;
    //     }
    //     if (i == -1)
    //     {
    //         return -1;
    //     }
    //     int imin = i;
    //     int vmin = sim[i];
    //     i--;
    //     while (i >= 0)
    //     {
    //         if (ids[i] != -1 && sim[i] < vmin)
    //         {
    //             vmin = sim[i];
    //             imin = i;
    //         }
    //         i--;
    //     }
    //     if (vmin_out)
    //     {
    //         *vmin_out = vmin;
    //     }
    //     int ret = ids[imin];
    //     ids[imin] = -1;
    //     --nvalid;

    //     return ret;
    // }

#endif

    int HNSW::MinimaxHeap::count_below(int thresh)
    {
        int n_below = 0;
        for (int i = 0; i < k; i++)
        {
            if (sim[i] < thresh)
            {
                n_below++;
            }
        }

        return n_below;
    }

} // namespace IFlib