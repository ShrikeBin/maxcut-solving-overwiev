#ifndef LOCAL_SEARCH_HPP
#define LOCAL_SEARCH_HPP

#include "Graph.hpp"
#include "CutState.hpp"
#include <cstdint>
#include <vector>

enum class LocalSearchMode {
    NAIVE_ONE_FLIP,
    K_FLIP,
    KERNIGHAN_LIN
};

struct LocalSearchResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

LocalSearchResult local_search(
    const Graph& graph,
    uint64_t seed = 42,
    const std::vector<int8_t>* initial_partition = nullptr,
    LocalSearchMode mode = LocalSearchMode::NAIVE_ONE_FLIP,
    int k = 2
);

#endif // LOCAL_SEARCH_HPP