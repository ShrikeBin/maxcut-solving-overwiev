#ifndef CUTSTATE_HPP
#define CUTSTATE_HPP

#include "Graph.hpp"

#include <vector>
#include <cstdint>

class CutState {
public:

    explicit CutState(const Graph& graph);

    // Generate random initial partition
    void randomize(uint64_t seed);

    // Flip vertex v
    void flip(int v);

    // Calculate everything from scratch
    void calculate();

    // Getters
    double cut_weight() const;
    int cut_edge_count() const;

    const std::vector<int8_t>& partition() const;
    const std::vector<double>& delta() const;

    // Change in cut weight if v is flipped
    double flip_delta(int v) const;

    // Initial state
    void set_partition(const std::vector<int8_t>& p);

private:

    const Graph& graph;

    std::vector<int8_t> partition_;
    std::vector<double> delta_;

    double cut_weight_ = 0.0;
    int cut_edge_count_ = 0;
};

#endif