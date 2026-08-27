#include "Approximations.hpp"
#include "CutState.hpp"
#include <random>
#include <algorithm>
#include <limits>

ApproxResult randomized_half(const Graph& graph, int trials, uint64_t seed) {
    CutState state(graph);
    std::mt19937_64 rng(seed);
    
    double best_weight = -std::numeric_limits<double>::infinity();
    int best_edge_count = 0;
    std::vector<int8_t> best_partition;

    for (int t = 0; t < trials; ++t) {
        state.randomize(rng());
        if (state.cut_weight() > best_weight) {
            best_weight = state.cut_weight();
            best_edge_count = state.cut_edge_count();
            best_partition = state.partition();
        }
    }

    return { best_partition, best_weight, best_edge_count };
}

ApproxResult randomized_greedy_edges(const Graph& graph, int trials, uint64_t seed) {
    int n = graph.vertex_count();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    // Get the base edges directly from the Graph class
    const auto& base_edges = graph.edges();

    double best_weight = -std::numeric_limits<double>::infinity();
    int best_edge_count = 0;
    std::vector<int8_t> best_partition;

    for (int t = 0; t < trials; ++t) {
        auto edges = base_edges;
        std::shuffle(edges.begin(), edges.end(), rng);

        std::vector<int8_t> partition(n, -1);
        for (const auto& e : edges) {
            int8_t p_u = partition[e.u];
            int8_t p_v = partition[e.v];

            if (p_u == -1 && p_v == -1) {
                int side = coin(rng);
                partition[e.u] = side;
                partition[e.v] = 1 - side;
            } else if (p_u != -1 && p_v == -1) {
                partition[e.v] = 1 - p_u;
            } else if (p_u == -1 && p_v != -1) {
                partition[e.u] = 1 - p_v;
            }
        }

        // Assign unassigned vertices randomly
        for (int i = 0; i < n; ++i) {
            if (partition[i] == -1) {
                partition[i] = coin(rng);
            }
        }

        CutState state(graph);
        state.set_partition(partition);

        if (state.cut_weight() > best_weight) {
            best_weight = state.cut_weight();
            best_edge_count = state.cut_edge_count();
            best_partition = partition;
        }
    }

    return { best_partition, best_weight, best_edge_count };
}

ApproxResult randomized_greedy_vertices(const Graph& graph, int trials, uint64_t seed) {
    int n = graph.vertex_count();
    std::mt19937_64 rng(seed);

    // Fast neighbor lookup via graph.adjacency()
    const auto& adj = graph.adjacency();

    double best_weight = -std::numeric_limits<double>::infinity();
    int best_edge_count = 0;
    std::vector<int8_t> best_partition;

    std::vector<int> nodes(n);
    for (int i = 0; i < n; ++i) nodes[i] = i;

    CutState state(graph);

    for (int t = 0; t < trials; ++t) {
        state.randomize(rng());
        std::shuffle(nodes.begin(), nodes.end(), rng);

        for (int u : nodes) {
            int8_t current_partition = state.partition()[u];
            double weight_internal = 0.0;
            double weight_crossing = 0.0;

            for (const auto& neighbor_pair : adj[u]) {
                int v = neighbor_pair.first;
                double weight = neighbor_pair.second;

                if (state.partition()[v] == current_partition) {
                    weight_internal += weight;
                } else {
                    weight_crossing += weight;
                }
            }

            if (weight_internal > weight_crossing) {
                state.flip(u);
            }
        }

        if (state.cut_weight() > best_weight) {
            best_weight = state.cut_weight();
            best_edge_count = state.cut_edge_count();
            best_partition = state.partition();
        }
    }

    return { best_partition, best_weight, best_edge_count };
}