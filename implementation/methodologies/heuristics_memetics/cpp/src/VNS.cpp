#include "VNS.hpp"
#include "LocalSearch.hpp"
#include <random>
#include <algorithm>
#include <unordered_set>
#include <omp.h>

// Fast O(k) dynamic perturbation for VNS shaking phase without n-sized allocations
static void shake(CutState& state, const Graph& graph, int k, std::mt19937_64& rng) {
    const int n = graph.vertex_count();
    if (n == 0) return;
    
    k = std::min(k, n);
    std::unordered_set<int> picked;
    picked.reserve(k);

    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(0, n - 1);
        int v = dist(rng);

        // Linear probing to resolve collisions in O(1) expected time
        while (picked.count(v)) {
            v = (v + 1) % n;
        }

        picked.insert(v);
        state.flip(v);
    }
}

// ----------------------------------------------------------------------------
// VARIABLE NEIGHBORHOOD SEARCH
// ----------------------------------------------------------------------------
VNSResult variable_neighborhood_search(
    const Graph& graph,
    uint64_t seed,
    const std::vector<int8_t>* initial_partition,
    int k_max,
    int max_iterations
) {
    CutState state(graph);
    if (initial_partition != nullptr) {
        state.set_partition(*initial_partition);
    } else {
        state.randomize(seed);
    }

    std::mt19937_64 rng(seed);

    auto initial_opt = local_search(graph, seed, &state.partition(), LocalSearchMode::KERNIGHAN_LIN);
    state.set_partition(initial_opt.partition);

    double best_weight = state.cut_weight();
    std::vector<int8_t> best_partition = state.partition();

    for (int iter = 0; iter < max_iterations; ++iter) {
        int k = 1;

        while (k <= k_max) {
            CutState work_state = state;

            // 1. Shake state by k flips (Optimized O(k))
            shake(work_state, graph, k, rng);

            // 2. Local Search refinement
            auto ls_res = local_search(graph, rng(), &work_state.partition(), LocalSearchMode::KERNIGHAN_LIN);
            
            // 3. Move or Expand Neighborhood
            if (ls_res.cut_weight > best_weight + 1e-9) {
                best_weight = ls_res.cut_weight;
                best_partition = ls_res.partition;
                state.set_partition(best_partition);
                k = 1;
            } else {
                k++;
            }
        }
    }

    state.set_partition(best_partition);
    return { best_partition, best_weight, state.cut_edge_count() };
}

// ----------------------------------------------------------------------------
// GREEDY RANDOMIZED ADAPTIVE SEARCH PROCEDURE (GRASP)
// ----------------------------------------------------------------------------
static std::vector<int8_t> build_rcl_partition(const Graph& graph, double alpha, std::mt19937_64& rng) {
    const int n = graph.vertex_count();
    std::vector<int8_t> partition(n, -1);
    CutState state(graph);

    partition[0] = rng() % 2;
    state.set_partition(partition);

    for (int step = 1; step < n; ++step) {
        double min_delta = 1e18, max_delta = -1e18;
        std::vector<double> deltas(n, 0.0);

        for (int v = 0; v < n; ++v) {
            if (partition[v] != -1) continue;
            double d = state.flip_delta(v);
            deltas[v] = d;
            min_delta = std::min(min_delta, d);
            max_delta = std::max(max_delta, d);
        }

        double threshold = max_delta - alpha * (max_delta - min_delta);
        std::vector<int> rcl;
        for (int v = 0; v < n; ++v) {
            if (partition[v] == -1 && deltas[v] >= threshold) {
                rcl.push_back(v);
            }
        }

        int selected_v = rcl[rng() % rcl.size()];
        partition[selected_v] = (deltas[selected_v] > 0) ? 1 : 0;
        state.set_partition(partition);
    }

    return partition;
}

GRASPResult grasp_search(const Graph& graph, uint64_t seed, int max_iterations, double alpha) {
    const int n = graph.vertex_count();
    double global_best_weight = -1e18;
    std::vector<int8_t> global_best_partition(n);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(seed + tid * 9999);

        double local_best_weight = -1e18;
        std::vector<int8_t> local_best_partition(n);

        #pragma omp for schedule(dynamic)
        for (int iter = 0; iter < max_iterations; ++iter) {
            std::vector<int8_t> rcl_part = build_rcl_partition(graph, alpha, rng);
            auto ls_res = local_search(graph, rng(), &rcl_part, LocalSearchMode::KERNIGHAN_LIN);

            if (ls_res.cut_weight > local_best_weight) {
                local_best_weight = ls_res.cut_weight;
                local_best_partition = ls_res.partition;
            }
        }

        #pragma omp critical
        {
            if (local_best_weight > global_best_weight) {
                global_best_weight = local_best_weight;
                global_best_partition = local_best_partition;
            }
        }
    }

    CutState final_state(graph);
    final_state.set_partition(global_best_partition);
    return { global_best_partition, global_best_weight, final_state.cut_edge_count() };
}