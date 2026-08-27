#include "TabuSearch.hpp"
#include <random>
#include <algorithm>
#include <vector>

TabuResult tabu_search(
    const Graph& graph,
    uint64_t seed,
    const std::vector<int8_t>* initial_partition,
    int tabu_tenure,
    int max_iterations,
    TabuMode mode
) {
    CutState state(graph);
    if (initial_partition != nullptr) {
        state.set_partition(*initial_partition);
    } else {
        state.randomize(seed);
    }

    const int n = graph.vertex_count();
    double current_weight = state.cut_weight();
    double best_weight = current_weight;
    std::vector<int8_t> best_partition = state.partition();

    // Memory structures
    std::vector<int> tabu_until(n, 0);       // Short-term recency memory
    std::vector<int> flip_frequency(n, 0);   // Long-term frequency memory

    for (int iter = 1; iter <= max_iterations; ++iter) {
        int best_v = -1;
        double best_score = -1e18;
        double chosen_delta = 0.0;

        for (int v = 0; v < n; ++v) {
            double delta = state.flip_delta(v);
            bool is_tabu = (tabu_until[v] >= iter);

            // Aspiration Criterion: Override tabu status if move yields new global best
            if (is_tabu && (current_weight + delta <= best_weight + 1e-9)) {
                continue;
            }

            // Long-term memory penalty for frequently flipped nodes
            double penalty = (mode == TabuMode::SHORT_AND_LONG_TERM) ? (0.01 * flip_frequency[v]) : 0.0;
            double move_score = delta - penalty;

            if (move_score > best_score) {
                best_score = move_score;
                chosen_delta = delta;
                best_v = v;
            }
        }

        // Defensive fallback: if all valid moves are tabu, pick the move with maximum raw delta
        if (best_v == -1) {
            for (int v = 0; v < n; ++v) {
                double delta = state.flip_delta(v);
                if (delta > best_score) {
                    best_score = delta;
                    chosen_delta = delta;
                    best_v = v;
                }
            }
        }

        if (best_v == -1) break;

        // Execute state move
        state.flip(best_v);
        current_weight += chosen_delta;
        tabu_until[best_v] = iter + tabu_tenure;
        flip_frequency[best_v]++;

        if (current_weight > best_weight + 1e-9) {
            best_weight = current_weight;
            best_partition = state.partition();
        }
    }

    state.set_partition(best_partition);
    return { best_partition, best_weight, state.cut_edge_count() };
}