#include "SimulatedAnnealing.hpp"
#include <random>
#include <cmath>

SAResult simulated_annealing(
    const Graph& graph,
    uint64_t seed,
    const std::vector<int8_t>* initial_partition,
    double initial_temp,
    double cooling_rate,
    int max_iterations
) {
    CutState state(graph);

    if (initial_partition != nullptr) {
        state.set_partition(*initial_partition);
    } else {
        state.randomize(seed);
    }

    double best_weight = state.cut_weight();
    int best_edge_count = state.cut_edge_count();
    std::vector<int8_t> best_partition = state.partition();

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_real(0.0, 1.0);
    std::uniform_int_distribution<int> dist_node(0, graph.vertex_count() - 1);

    double temp = initial_temp;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int v = dist_node(rng);
        double delta = state.flip_delta(v);

        if (delta > 0 || dist_real(rng) < std::exp(delta / temp)) {
            state.flip(v);
            if (state.cut_weight() > best_weight) {
                best_weight = state.cut_weight();
                best_edge_count = state.cut_edge_count();
                best_partition = state.partition();
            }
        }
        temp *= cooling_rate;
        if (temp < 1e-6) break;
    }

    return { best_partition, best_weight, best_edge_count };
}