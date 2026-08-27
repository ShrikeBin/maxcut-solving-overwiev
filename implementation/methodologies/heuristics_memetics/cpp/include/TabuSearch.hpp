#ifndef TABU_SEARCH_HPP
#define TABU_SEARCH_HPP

#include "Graph.hpp"
#include "CutState.hpp"
#include <vector>
#include <cstdint>

struct TabuResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

enum class TabuMode {
    SHORT_TERM_ONLY,
    SHORT_AND_LONG_TERM // Includes Frequency-Based Diversification
};

TabuResult tabu_search(
    const Graph& graph,
    uint64_t seed = 42,
    const std::vector<int8_t>* initial_partition = nullptr,
    int tabu_tenure = 10,
    int max_iterations = 10000,
    TabuMode mode = TabuMode::SHORT_TERM_ONLY
);

#endif // TABU_SEARCH_HPP