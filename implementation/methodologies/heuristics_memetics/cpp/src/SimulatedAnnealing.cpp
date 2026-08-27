#include "SimulatedAnnealing.hpp"
#include <random>
#include <cmath>
#include <vector>

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

    const int n = graph.vertex_count();
    
    // Track best cut stats (avoid vector copies inside the loop)
    double current_weight = state.cut_weight();
    double best_weight = current_weight;
    std::vector<int8_t> best_partition = state.partition();

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_real(0.0, 1.0);
    std::uniform_int_distribution<int> dist_node(0, n - 1);

    double temp = initial_temp;

    // Perform an inner sweep of moves (e.g., n steps) per temperature level
    int moves_per_temp = n; 
    int outer_steps = max_iterations / moves_per_temp;
    if (outer_steps < 1) outer_steps = 1;

    for (int iter = 0; iter < outer_steps; ++iter) {
        for (int step = 0; step < moves_per_temp; ++step) {
            int v = dist_node(rng);
            double delta = state.flip_delta(v);

            // Metropolis Criterion for MaxCut
            if (delta > 0.0 || dist_real(rng) < std::exp(delta / temp)) {
                state.flip(v);
                current_weight += delta;

                if (current_weight > best_weight) {
                    best_weight = current_weight;
                    // Copy partition ONLY when a new global best is found
                    best_partition = state.partition(); 
                }
            }
        }

        temp *= cooling_rate;
        if (temp < 1e-12) break; // Lower threshold to allow deep cooling
    }

    // Reconstruct final best edge count from saved state
    state.set_partition(best_partition);

    return { best_partition, best_weight, state.cut_edge_count() };
}