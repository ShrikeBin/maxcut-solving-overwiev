#ifndef LOCAL_SEARCH_HPP
#define LOCAL_SEARCH_HPP

#include "Graph.hpp"
#include "CutState.hpp"
#include <cstdint>
#include <vector>

struct LocalSearchResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

LocalSearchResult local_search(
    const Graph& graph,
    uint64_t seed = 42,
    const std::vector<int8_t>* initial_partition = nullptr
);

#endif // LOCAL_SEARCH_HPP