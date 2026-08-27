#include <cstdint>
#include <vector>
#include "Graph.hpp"
#include "LocalSearch.hpp"
#include "SimulatedAnnealing.hpp"
#include "TabuSearch.hpp"
#include "Evolutionary.hpp"
#include "VNS.hpp"
#include "Approximations.hpp"

// Standard C Return Struct matching Python ctypes definition
extern "C" {
    struct CResult {
        double cut_weight;
        int cut_edge_count;
        int8_t* partition;
    };

    // Memory cleanup for returned partitions
    void free_c_result(CResult res) {
        if (res.partition != nullptr) {
            delete[] res.partition;
        }
    }
}

// Helper to convert C++ vector partition to CResult struct
static CResult make_c_result(int n, double weight, int edge_count, const std::vector<int8_t>& part) {
    int8_t* c_part = new int8_t[n];
    for (int i = 0; i < n; ++i) {
        c_part[i] = part[i];
    }
    return CResult{weight, edge_count, c_part};
}

extern "C" {

    // ==========================================
    // LOCAL SEARCH
    // ==========================================
    CResult run_local_search(
        const char* filename,
        uint64_t seed,
        const int8_t* initial_partition,
        int mode_int,
        int k
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();
        LocalSearchMode mode = static_cast<LocalSearchMode>(mode_int);

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = local_search(graph, seed, init_ptr, mode, k);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    // ==========================================
    // SIMULATED ANNEALING
    // ==========================================
    CResult run_simulated_annealing(
        const char* filename,
        uint64_t seed,
        const int8_t* initial_partition,
        double initial_temp,
        double cooling_rate,
        int max_iterations
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = simulated_annealing(graph, seed, init_ptr, initial_temp, cooling_rate, max_iterations);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    // ==========================================
    // TABU SEARCH
    // ==========================================
    CResult run_tabu_search(
        const char* filename,
        uint64_t seed,
        const int8_t* initial_partition,
        int tabu_tenure,
        int max_iterations,
        int mode_int
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();
        TabuMode mode = static_cast<TabuMode>(mode_int);

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = tabu_search(graph, seed, init_ptr, tabu_tenure, max_iterations, mode);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    // ==========================================
    // EVOLUTIONARY & MEMETIC ALGORITHMS (UPDATED)
    // ==========================================
    CResult run_genetic_algorithm(
        const char* filename,
        uint64_t seed,
        int population_size,
        int generations,
        double mutation_rate,
        const int8_t* initial_partition
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = genetic_algorithm(graph, seed, population_size, generations, mutation_rate, init_ptr);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    CResult run_genetic_kl_algorithm(
        const char* filename,
        uint64_t seed,
        int population_size,
        int generations,
        double mutation_rate,
        const int8_t* initial_partition
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = genetic_KL_algorithm(graph, seed, population_size, generations, mutation_rate, init_ptr);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    CResult run_island_kl_algorithm(
        const char* filename,
        uint64_t seed,
        int population_size,
        int generations,
        int migration_interval,
        double mutation_rate,
        const int8_t* initial_partition
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = island_kl_algorithm(graph, seed, population_size, generations, migration_interval, mutation_rate, init_ptr);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    // ==========================================
    // VARIABLE NEIGHBORHOOD SEARCH
    // ==========================================
    CResult run_vns(
        const char* filename,
        uint64_t seed,
        const int8_t* initial_partition,
        int k_max,
        int max_iterations
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();

        std::vector<int8_t> init_part;
        const std::vector<int8_t>* init_ptr = nullptr;
        if (initial_partition != nullptr) {
            init_part.assign(initial_partition, initial_partition + n);
            init_ptr = &init_part;
        }

        auto res = variable_neighborhood_search(graph, seed, init_ptr, k_max, max_iterations);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    // ==========================================
    // GRASP
    // ==========================================
    CResult run_grasp(
        const char* filename,
        uint64_t seed,
        int max_iterations,
        double alpha
    ) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = grasp_search(graph, seed, max_iterations, alpha);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    // ==========================================
    // APPROXIMATIONS
    // ==========================================
    CResult run_randomized_half(const char* filename, int trials, uint64_t seed) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = randomized_half(graph, trials, seed);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    CResult run_randomized_greedy_edges(const char* filename, int trials, uint64_t seed) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = randomized_greedy_edges(graph, trials, seed);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }

    CResult run_randomized_greedy_vertices(const char* filename, int trials, uint64_t seed) {
        Graph graph(filename);
        int n = graph.vertex_count();
        auto res = randomized_greedy_vertices(graph, trials, seed);
        return make_c_result(n, res.cut_weight, res.cut_edge_count, res.partition);
    }
}