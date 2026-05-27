#pragma once

/*
This implementation is based on the FAISS project: https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h
*/


#include <queue>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <cstddef>

#include <omp.h>

#ifdef USE_AVX2
#include "tools/SimComputer_avx2.h"
#else
#include "tools/SimComputer.h"
#endif

#include "tools/RandomGenerator.h"
#include "tools/AuxStructures.h"
#include "tools/ResHandler.h"

namespace IFlib
{

struct HNSWStats
{
    size_t n1 = 0; /// number of vectors searched
    size_t n2 =
        0;            /// number of queries for which the candidate list is exhausted
    size_t ndis = 0;  /// number of distances computed
    size_t nhops = 0; /// number of hops aka number of edges traversed

    void reset()
    {
        n1 = n2 = 0;
        ndis = 0;
        nhops = 0;
    }

    void combine(const HNSWStats &other)
    {
        n1 += other.n1;
        n2 += other.n2;
        ndis += other.ndis;
        nhops += other.nhops;
    }
};

struct HNSW
{
    using storage_idx_t = int32_t;

    struct MinimaxHeap
    {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<int> sim;
        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), sim(n) {}

        void push(storage_idx_t i, int v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(int *vmin_out = nullptr);

        int count_below(int thresh);
    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser
    {
        int sim;
        int id;
        NodeDistCloser(int sim, int id) : sim(sim), id(id) {}
        bool operator<(const NodeDistCloser &obj1) const
        {
            return sim < obj1.sim;
        }
    };

    struct NodeDistFarther
    {
        int sim;
        int id;
        NodeDistFarther(int sim, int id) : sim(sim), id(id) {}
        bool operator<(const NodeDistFarther &obj1) const
        {
            return sim > obj1.sim;
        }
    };

    /// assignment probability to each layer (sum=1)
    std::vector<double> assign_probas;

    /// number of neighbors stored per layer (cumulative), should not
    /// be changed after first add
    std::vector<int> cum_nneighbor_per_level;

    /// level of each vector (base level = 1), size = ntotal
    std::vector<int> levels;

    /// offsets[i] is the offset in the neighbors array where vector i is stored
    /// size ntotal + 1
    std::vector<size_t> offsets;

    /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    /// for all levels. this is where all storage goes.
    std::vector<storage_idx_t> neighbors;

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point = -1;

    IFlib::RandomGenerator rng;

    /// maximum level
    int max_level = -1;

    /// expansion factor at construction time
    int efConstruction = 40;

    /// expansion factor at search time
    int efSearch = 16;

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

    // methods that initialize the tree sizes

    /// initialize the assign_probas and cum_nneighbor_per_level to
    /// have 2*M links on level 0 and M links on levels > 0
    void set_default_probas(int M, float levelMult);

    /// set nb of neighbors for this level (before adding anything)
    void set_nb_neighbors(int level_no, int n);

    // methods that access the tree sizes
    /// nb of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    /// range of entries in the neighbors table of vertex no at layer_no
    void neighbor_range(size_t no, int layer_no, size_t *begin, size_t *end)
        const;

    /// only mandatory parameter: nb of neighbors
    explicit HNSW(int M = 32);

    /// pick a random level for a new point
    int random_level();

    // Find nearest neighbors in the specified layer and update neighbors
    void add_links_starting_from(
        IForestSimComputer &ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        int d_nearest,
        int level,
        omp_lock_t *locks,
        VisitedTable &vt,
        bool keep_max_size_level0 = false);

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(
        IForestSimComputer &ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t> &locks,
        VisitedTable &vt,
        bool keep_max_size_level0 = false);

    // search interface for 1 point, single thread
    HNSWStats search(
        IForestSimComputer &qdis,
        BatchMinHeapHandler::SingleResultHandler &res,
        VisitedTable &vt) const;


    void reset();

    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;

    // pre-allocate index memory
    int prepare_level_tab(size_t n, bool preset_levels = false);

    static void shrink_neighbor_list(
        IForestSimComputer &qdis,
        std::priority_queue<NodeDistCloser> &input,
        std::vector<NodeDistCloser> &output,
        int max_size,
        bool keep_max_size_level0 = false);

    void permute_entries(const size_t *map);
};

// Find the entry point in the specified layer
HNSWStats greedy_update_nearest(
    const HNSW &hnsw,
    IForestSimComputer &qdis,
    int level,
    HNSW::storage_idx_t &nearest,
    int &d_nearest);

// Find the nearest neighbor (entry point) in the specified layer
void search_neighbors_to_add(
    HNSW &hnsw,
    IForestSimComputer &qdis,
    std::priority_queue<HNSW::NodeDistFarther> &results,
    int entry_point,
    int d_entry_point,
    int level,
    VisitedTable &vt);

void search_from_candidates(
    const HNSW &hnsw,
    IForestSimComputer &qdis,
    BatchMinHeapHandler::SingleResultHandler &res,
    HNSW::MinimaxHeap &candidates,
    VisitedTable &vt,
    HNSWStats &stats,
    int level);

} // namespace IFlib
