#ifndef SIMULATED_ANNEALING_HPP
#define SIMULATED_ANNEALING_HPP

#include "Graph.hpp"
#include "CutState.hpp"
#include <cstdint>
#include <vector>

struct SAResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

SAResult simulated_annealing(
    const Graph& graph,
    uint64_t seed = 42,
    const std::vector<int8_t>* initial_partition = nullptr,
    double initial_temp = 100.0,
    double cooling_rate = 0.995,
    int max_iterations = 100000
);

#endif // SIMULATED_ANNEALING_HPP