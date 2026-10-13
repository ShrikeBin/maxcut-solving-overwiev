#include <cstdint>
#include <vector>
#include "Graph.hpp"
#include "Approximations.hpp"

// Standard C Return Struct matching Python ctypes definition
extern "C" {
    struct CResult {
        double cut_weight;
        int cut_edge_count;
        int8_t* partition;
    };

    // Memory cleanup for returned partitions
    void free_c_result(CResult res) {
        if (res.partition != nullptr) {
            delete[] res.partition;
        }
    }
}

// Helper to convert C++ vector partition to CResult struct
static CResult make_c_result(int n, double weight, int edge_count, const std::vector<int8_t>& part) {
    int8_t* c_part = new int8_t[n];
    for (int i = 0; i < n; ++i) {
        c_part[i] = part[i];
    }
    return CResult{weight, edge_count, c_part};
}

extern "C" {
    // ==========================================
    // APPROXIMATIONS
    // ==========================================
    CResult run_randomized_half(const char* filename, int trials, uint64_t seed) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = randomized_half(graph, trials, seed);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    CResult run_randomized_greedy_edges(const char* filename, int trials, uint64_t seed) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = randomized_greedy_edges(graph, trials, seed);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    CResult run_randomized_greedy_vertices(const char* filename, int trials, uint64_t seed) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = randomized_greedy_vertices(graph, trials, seed);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }
}