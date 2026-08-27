#ifndef VNS_HPP
#define VNS_HPP

#include "Graph.hpp"
#include "CutState.hpp"
#include <vector>
#include <cstdint>

struct VNSResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

struct GRASPResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

// Variable Neighborhood Search (VNS)
VNSResult variable_neighborhood_search(
    const Graph& graph,
    uint64_t seed = 42,
    const std::vector<int8_t>* initial_partition = nullptr,
    int k_max = 4,
    int max_iterations = 1000
);

// Greedy Randomized Adaptive Search Procedure (GRASP)
GRASPResult grasp_search(
    const Graph& graph,
    uint64_t seed = 42,
    int max_iterations = 50,
    double alpha = 0.3
);

#endif // VNS_HPP