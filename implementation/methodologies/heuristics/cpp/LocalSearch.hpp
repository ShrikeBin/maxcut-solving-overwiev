#ifndef LOCALSEARCH_HPP
#define LOCALSEARCH_HPP

#include "../../graphs/cpp/Graph.hpp"
#include "../../graphs/cpp/CutState.hpp"

struct LocalSearchResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

LocalSearchResult local_search(
    const Graph& graph,
    uint64_t seed
);

#endif