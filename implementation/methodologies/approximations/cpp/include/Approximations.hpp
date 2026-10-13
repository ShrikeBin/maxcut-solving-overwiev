#ifndef APPROXIMATIONS_HPP
#define APPROXIMATIONS_HPP

#include "Graph.hpp"
#include <vector>
#include <cstdint>

struct ApproxResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

ApproxResult randomized_half(const Graph& graph, int trials, uint64_t seed = 42);
ApproxResult randomized_greedy_edges(const Graph& graph, int trials, uint64_t seed = 42);
ApproxResult randomized_greedy_vertices(const Graph& graph, int trials, uint64_t seed = 42);

#endif // APPROXIMATIONS_HPP