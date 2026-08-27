#include "LocalSearch.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>

LocalSearchResult local_search(
    const Graph& graph,
    uint64_t seed,
    const std::vector<int8_t>* initial_partition,
    LocalSearchMode mode,
    int k
) {
    CutState state(graph);

    if (initial_partition != nullptr) {
        state.set_partition(*initial_partition);
    } else {
        state.randomize(seed);
    }

    const int n = graph.vertex_count();

    // ------------------------------------------------------------------------
    // MODE 1: NAIVE 1-FLIP (Steepest Descent)
    // ------------------------------------------------------------------------
    if (mode == LocalSearchMode::NAIVE_ONE_FLIP) {
        while (true) {
            int best_vertex = -1;
            double best_delta = 0.0;

            for (int v = 0; v < n; ++v) {
                double delta = state.flip_delta(v);
                if (delta > best_delta) {
                    best_delta = delta;
                    best_vertex = v;
                }
            }

            if (best_vertex == -1) break;
            state.flip(best_vertex);
        }
    }

    // ------------------------------------------------------------------------
    // MODE 2: K-FLIP NEIGHBORHOOD (OpenMP for k-opt)
    // ------------------------------------------------------------------------
    else if (mode == LocalSearchMode::K_FLIP) {
        bool global_improved = true;

        while (global_improved) {
            global_improved = false;

            // 1. Exhaust 1-flips sequentially for speed
            int best_v = -1;
            double best_d1 = 0.0;
            for (int v = 0; v < n; ++v) {
                double d = state.flip_delta(v);
                if (d > best_d1) {
                    best_d1 = d;
                    best_v = v;
                }
            }

            if (best_v != -1) {
                state.flip(best_v);
                global_improved = true;
                continue;
            }

            // Parallelize searching combinations for depths 2 to k
            for (int depth = 2; depth <= k; ++depth) {
                std::vector<int> global_best_combo;
                double global_max_delta = 0.0;

                #pragma omp parallel
                {
                    CutState local_state = state; 
                    std::vector<int> local_best_combo;
                    double local_max_delta = 0.0;

                    auto find_k_flip = [&](auto& self, int start_idx, int current_depth, int max_depth, 
                                           double accumulated_delta, std::vector<int>& current_combo) -> void {
                        
                        for (int v = start_idx; v < n; ++v) {
                            double delta_v = local_state.flip_delta(v);
                            double total_delta = accumulated_delta + delta_v;

                            current_combo.push_back(v);
                            local_state.flip(v);

                            if (total_delta > local_max_delta + 1e-9) {
                                local_max_delta = total_delta;
                                local_best_combo = current_combo;
                            }

                            if (current_depth < max_depth) {
                                self(self, v + 1, current_depth + 1, max_depth, total_delta, current_combo);
                            }

                            local_state.flip(v);
                            current_combo.pop_back();
                        }
                    };

                    #pragma omp for schedule(dynamic)
                    for (int start_v = 0; start_v < n; ++start_v) {
                        double delta_v = local_state.flip_delta(start_v);
                        std::vector<int> current_combo = { start_v };
                        
                        local_state.flip(start_v);

                        if (delta_v > local_max_delta + 1e-9) {
                            local_max_delta = delta_v;
                            local_best_combo = current_combo;
                        }

                        if (1 < depth) {
                            find_k_flip(find_k_flip, start_v + 1, 2, depth, delta_v, current_combo);
                        }

                        local_state.flip(start_v);
                    }

                    #pragma omp critical
                    {
                        if (local_max_delta > global_max_delta + 1e-9) {
                            global_max_delta = local_max_delta;
                            global_best_combo = local_best_combo;
                        }
                    }
                }

                if (!global_best_combo.empty() && global_max_delta > 0.0) {
                    for (int v : global_best_combo) {
                        state.flip(v);
                    }
                    global_improved = true;
                    break; // Apply best move and fall back to 1-flip phase
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // MODE 3: KERNIGHAN-LIN (FM-Style Variable-Depth Pass)
    // ------------------------------------------------------------------------
    else if (mode == LocalSearchMode::KERNIGHAN_LIN) {
        std::vector<bool> locked(n, false);
        bool outer_improved = true;

        while (outer_improved) {
            outer_improved = false;
            std::fill(locked.begin(), locked.end(), false);

            double current_weight = state.cut_weight();
            double max_pass_weight = current_weight;
            int best_step = -1;

            std::vector<int> flip_sequence;
            flip_sequence.reserve(n);

            for (int step = 0; step < n; ++step) {
                int best_vertex = -1;
                double max_delta = -1e18;

                for (int v = 0; v < n; ++v) {
                    if (locked[v]) continue;

                    double delta = state.flip_delta(v);
                    if (delta > max_delta) {
                        max_delta = delta;
                        best_vertex = v;
                    }
                }

                if (best_vertex == -1) break;

                state.flip(best_vertex);
                locked[best_vertex] = true;
                flip_sequence.push_back(best_vertex);
                current_weight += max_delta;

                if (current_weight > max_pass_weight + 1e-9) {
                    max_pass_weight = current_weight;
                    best_step = step;
                }
            }

            int total_flips = static_cast<int>(flip_sequence.size());
            for (int i = total_flips - 1; i > best_step; --i) {
                state.flip(flip_sequence[i]);
            }

            if (best_step != -1) {
                outer_improved = true;
            }
        }
    }

    return { state.partition(), state.cut_weight(), state.cut_edge_count() };
}