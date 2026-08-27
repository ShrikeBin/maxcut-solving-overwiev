#include "Graph.hpp"
#include "LocalSearch.hpp"
#include "SimulatedAnnealing.hpp"
#include "Approximations.hpp"
#include <vector>
#include <cstdint>

extern "C" {

struct CResult {
    double cut_weight;
    int cut_edge_count;
    int* partition;
};

void free_c_result(CResult res) {
    if (res.partition != nullptr) {
        delete[] res.partition;
    }
}

static CResult make_c_result(int n, double weight, int edge_count, const std::vector<int8_t>& part) {
    int* out_part = new int[n];
    for (int i = 0; i < n; ++i) {
        out_part[i] = part[i];
    }
    return CResult{ weight, edge_count, out_part };
}

// ==========================================
// Heuristics
// ==========================================

CResult run_local_search(
    const char* filename,
    uint64_t seed,
    const int8_t* initial_partition,
    int mode_int,
    int k
) {
    Graph graph(filename);
    int n = graph.vertex_count();
    LocalSearchMode mode = static_cast<LocalSearchMode>(mode_int);

    if (initial_partition != nullptr) {
        std::vector<int8_t> init_part(initial_partition, initial_partition + n);
        auto res = local_search(graph, seed, &init_part, mode, k);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    } else {
        auto res = local_search(graph, seed, nullptr, mode, k);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }
}

CResult run_simulated_annealing(
    const char* filename,
    uint64_t seed,
    const int8_t* initial_partition,
    double initial_temp,
    double cooling_rate,
    int max_iterations
) {
    Graph graph(filename);
    int n = graph.vertex_count();

    if (initial_partition != nullptr) {
        std::vector<int8_t> init_part(initial_partition, initial_partition + n);
        auto res = simulated_annealing(graph, seed, &init_part, initial_temp, cooling_rate, max_iterations);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    } else {
        auto res = simulated_annealing(graph, seed, nullptr, initial_temp, cooling_rate, max_iterations);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }
}

// ==========================================
// Approximations
// ==========================================

CResult run_randomized_half(
    const char* filename,
    int trials,
    uint64_t seed
) {
    Graph graph(filename);
    int n = graph.vertex_count();
    auto res = randomized_half(graph, trials, seed);
    return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
}

CResult run_randomized_greedy_edges(
    const char* filename,
    int trials,
    uint64_t seed
) {
    Graph graph(filename);
    int n = graph.vertex_count();
    auto res = randomized_greedy_edges(graph, trials, seed);
    return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
}

CResult run_randomized_greedy_vertices(
    const char* filename,
    int trials,
    uint64_t seed
) {
    Graph graph(filename);
    int n = graph.vertex_count();
    auto res = randomized_greedy_vertices(graph, trials, seed);
    return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
}

} // extern "C"