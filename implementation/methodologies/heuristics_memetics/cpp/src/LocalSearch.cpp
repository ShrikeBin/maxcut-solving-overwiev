#include "LocalSearch.hpp"

LocalSearchResult local_search(
    const Graph& graph,
    uint64_t seed,
    const std::vector<int8_t>* initial_partition
) {
    CutState state(graph);

    if (initial_partition != nullptr) {
        state.set_partition(*initial_partition);
    } else {
        state.randomize(seed);
    }

    while (true) {
        int best_vertex = -1;
        double best_delta = 0.0;

        for (int v = 0; v < graph.vertex_count(); ++v) {
            double delta = state.flip_delta(v);
            if (delta > best_delta) {
                best_delta = delta;
                best_vertex = v;
            }
        }

        if (best_vertex == -1) break;
        state.flip(best_vertex);
    }

    return { state.partition(), state.cut_weight(), state.cut_edge_count() };
}