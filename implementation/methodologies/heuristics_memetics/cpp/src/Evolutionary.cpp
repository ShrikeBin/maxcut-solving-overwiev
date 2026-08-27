#include "Evolutionary.hpp"
#include "LocalSearch.hpp"
#include <random>
#include <algorithm>
#include <omp.h>

struct Individual {
    std::vector<int8_t> chromosome;
    double fitness;
};

// Uniform Crossover
static Individual crossover(const Individual& p1, const Individual& p2, std::mt19937_64& rng, const Graph& graph) {
    const int n = graph.vertex_count();
    std::vector<int8_t> child(n);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        child[i] = (dist(rng) < 0.5) ? p1.chromosome[i] : p2.chromosome[i];
    }

    CutState state(graph);
    state.set_partition(child);
    return { child, state.cut_weight() };
}

// Mutation
static void mutate(Individual& ind, double mutation_rate, std::mt19937_64& rng, const Graph& graph) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    bool changed = false;
    for (size_t i = 0; i < ind.chromosome.size(); ++i) {
        if (dist(rng) < mutation_rate) {
            ind.chromosome[i] = 1 - ind.chromosome[i];
            changed = true;
        }
    }
    if (changed) {
        CutState state(graph);
        state.set_partition(ind.chromosome);
        ind.fitness = state.cut_weight();
    }
}

// ----------------------------------------------------------------------------
// 1. PURE GENETIC ALGORITHM
// ----------------------------------------------------------------------------
EvolutionaryResult genetic_algorithm(
    const Graph& graph, uint64_t seed,
    int population_size, int generations, double mutation_rate,
    const std::vector<int8_t>* initial_partition
) {
    const int n = graph.vertex_count();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pop_dist(0, population_size - 1);

    std::vector<Individual> pop(population_size);
    int start_idx = 0;

    if (initial_partition != nullptr) {
        CutState state(graph);
        state.set_partition(*initial_partition);
        pop[0] = { *initial_partition, state.cut_weight() };
        start_idx = 1;
    }

    for (int i = start_idx; i < population_size; ++i) {
        std::vector<int8_t> part(n);
        for (int j = 0; j < n; ++j) part[j] = rng() % 2;
        CutState state(graph);
        state.set_partition(part);
        pop[i] = { part, state.cut_weight() };
    }

    for (int gen = 0; gen < generations; ++gen) {
        std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });

        int p1_idx = std::min(pop_dist(rng), pop_dist(rng));
        int p2_idx = std::min(pop_dist(rng), pop_dist(rng));

        Individual child = crossover(pop[p1_idx], pop[p2_idx], rng, graph);
        mutate(child, mutation_rate, rng, graph);

        if (child.fitness > pop.back().fitness) {
            pop.back() = child;
        }
    }

    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
    });

    CutState final_state(graph);
    final_state.set_partition(pop[0].chromosome);
    return { pop[0].chromosome, pop[0].fitness, final_state.cut_edge_count() };
}

// ----------------------------------------------------------------------------
// 2. MEMETIC ALGORITHM WITH KERNIGHAN-LIN (Single Threaded / Sequential)
// ----------------------------------------------------------------------------
EvolutionaryResult genetic_KL_algorithm(
    const Graph& graph, uint64_t seed,
    int population_size, int generations, double mutation_rate,
    const std::vector<int8_t>* initial_partition
) {
    const int n = graph.vertex_count();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pop_dist(0, population_size - 1);

    std::vector<Individual> pop(population_size);
    int start_idx = 0;

    if (initial_partition != nullptr) {
        auto kl_res = local_search(graph, rng(), initial_partition, LocalSearchMode::KERNIGHAN_LIN);
        pop[0] = { kl_res.partition, kl_res.cut_weight };
        start_idx = 1;
    }

    for (int i = start_idx; i < population_size; ++i) {
        std::vector<int8_t> part(n);
        for (int j = 0; j < n; ++j) part[j] = rng() % 2;

        auto kl_res = local_search(graph, rng(), &part, LocalSearchMode::KERNIGHAN_LIN);
        pop[i] = { kl_res.partition, kl_res.cut_weight };
    }

    for (int gen = 0; gen < generations; ++gen) {
        std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });

        int p1_idx = std::min(pop_dist(rng), pop_dist(rng));
        int p2_idx = std::min(pop_dist(rng), pop_dist(rng));

        Individual child = crossover(pop[p1_idx], pop[p2_idx], rng, graph);
        mutate(child, mutation_rate, rng, graph);

        // Kernighan-Lin Refinement
        auto kl_res = local_search(graph, rng(), &child.chromosome, LocalSearchMode::KERNIGHAN_LIN);
        child.chromosome = kl_res.partition;
        child.fitness = kl_res.cut_weight;

        if (child.fitness > pop.back().fitness) {
            pop.back() = child;
        }
    }

    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
    });

    CutState final_state(graph);
    final_state.set_partition(pop[0].chromosome);
    return { pop[0].chromosome, pop[0].fitness, final_state.cut_edge_count() };
}

// ----------------------------------------------------------------------------
// 3. ISLAND MODEL MEMETIC ALGORITHM (Parallel with Ring Migration)
// ----------------------------------------------------------------------------
EvolutionaryResult island_kl_algorithm(
    const Graph& graph, uint64_t seed,
    int population_size, int generations,
    int migration_interval, double mutation_rate,
    const std::vector<int8_t>* initial_partition
) {
    const int n = graph.vertex_count();
    int num_islands = omp_get_max_threads();
    int island_pop_size = std::max(4, population_size / num_islands);

    std::vector<std::vector<Individual>> islands(num_islands, std::vector<Individual>(island_pop_size));
    std::vector<Individual> migration_buffers(num_islands);

    #pragma omp parallel num_threads(num_islands)
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(seed + tid * 777);
        std::uniform_int_distribution<int> pop_dist(0, island_pop_size - 1);

        auto& pop = islands[tid];
        int start_idx = 0;

        if (initial_partition != nullptr) {
            auto kl_res = local_search(graph, rng(), initial_partition, LocalSearchMode::KERNIGHAN_LIN);
            pop[0] = { kl_res.partition, kl_res.cut_weight };
            start_idx = 1;
        }

        // Initialize local population with KL refinement
        for (int i = start_idx; i < island_pop_size; ++i) {
            std::vector<int8_t> part(n);
            for (int j = 0; j < n; ++j) part[j] = rng() % 2;
            
            auto kl_res = local_search(graph, rng(), &part, LocalSearchMode::KERNIGHAN_LIN);
            pop[i] = { kl_res.partition, kl_res.cut_weight };
        }

        for (int gen = 0; gen < generations; ++gen) {
            std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
            });

            // Handle Ring Migration at fixed intervals
            if (gen > 0 && gen % migration_interval == 0) {
                migration_buffers[tid] = pop[0]; // Export best individual to buffer

                #pragma omp barrier

                // Import elite individual from neighboring island (Ring topology)
                int source_island = (tid + num_islands - 1) % num_islands;
                pop.back() = migration_buffers[source_island];

                #pragma omp barrier
            }

            int p1_idx = std::min(pop_dist(rng), pop_dist(rng));
            int p2_idx = std::min(pop_dist(rng), pop_dist(rng));

            Individual child = crossover(pop[p1_idx], pop[p2_idx], rng, graph);
            mutate(child, mutation_rate, rng, graph);

            auto kl_res = local_search(graph, rng(), &child.chromosome, LocalSearchMode::KERNIGHAN_LIN);
            child.chromosome = kl_res.partition;
            child.fitness = kl_res.cut_weight;

            if (child.fitness > pop.back().fitness) {
                pop.back() = child;
            }
        }

        std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    }

    // Find global best across all islands
    Individual global_best{ {}, -1e18 };
    for (int i = 0; i < num_islands; ++i) {
        if (islands[i][0].fitness > global_best.fitness) {
            global_best = islands[i][0];
        }
    }

    CutState final_state(graph);
    final_state.set_partition(global_best.chromosome);
    return { global_best.chromosome, global_best.fitness, final_state.cut_edge_count() };
}