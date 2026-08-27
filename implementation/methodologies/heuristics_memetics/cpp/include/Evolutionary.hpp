#ifndef EVOLUTIONARY_HPP
#define EVOLUTIONARY_HPP

#include "Graph.hpp"
#include "CutState.hpp"
#include <vector>
#include <cstdint>

struct EvolutionaryResult {
    std::vector<int8_t> partition;
    double cut_weight;
    int cut_edge_count;
};

// 1. Pure Genetic Algorithm (No local search refinement)
EvolutionaryResult genetic_algorithm(
    const Graph& graph, uint64_t seed = 42,
    int population_size = 30, int generations = 100, double mutation_rate = 0.05,
    const std::vector<int8_t>* initial_partition = nullptr
);

// 2. Memetic Algorithm (GA + Kernighan-Lin refinement per child)
EvolutionaryResult genetic_KL_algorithm(
    const Graph& graph, uint64_t seed = 42,
    int population_size = 30, int generations = 100, double mutation_rate = 0.05,
    const std::vector<int8_t>* initial_partition = nullptr
);

// 3. Island Model Memetic Algorithm (Parallel isolated populations with periodic migration)
EvolutionaryResult island_kl_algorithm(
    const Graph& graph, uint64_t seed = 42,
    int population_size = 30, int generations = 100, 
    int migration_interval = 10, double mutation_rate = 0.05,
    const std::vector<int8_t>* initial_partition = nullptr
);

#endif // EVOLUTIONARY_HPP