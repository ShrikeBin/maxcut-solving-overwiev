#include "CutState.hpp"

#include <random>

CutState::CutState(const Graph& graph)
    : graph(graph),
      partition_(graph.vertex_count(), 0),
      delta_(graph.vertex_count(), 0.0) {
}

void CutState::randomize(uint64_t seed) {

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> bit(0, 1);

    for (auto& p : partition_)
        p = bit(rng);

    calculate();
}

void CutState::calculate() {

    cut_weight_ = 0.0;
    cut_edge_count_ = 0;

    std::fill(delta_.begin(), delta_.end(), 0.0);

    // Calculate current cut
    for (const auto& edge : graph.edges()) {

        if (partition_[edge.u] != partition_[edge.v]) {
            cut_weight_ += edge.weight;
            ++cut_edge_count_;
        }
    }

    // Calculate flip deltas
    for (int v = 0; v < graph.vertex_count(); ++v) {

        for (const auto& [u, weight] : graph.adjacency()[v]) {

            if (partition_[v] == partition_[u])
                delta_[v] += weight;
            else
                delta_[v] -= weight;
        }
    }
}

void CutState::flip(int v) {

    double change = delta_[v];

    // Determine how many incident edges change cut status.
    int edge_count_change = 0;

    for (const auto& [u, weight] : graph.adjacency()[v]) {

        bool currently_cut = partition_[v] != partition_[u];

        if (currently_cut)
            --edge_count_change;
        else
            ++edge_count_change;
    }

    partition_[v] = 1 - partition_[v];

    cut_weight_ += change;
    cut_edge_count_ += edge_count_change;

    delta_[v] = -delta_[v];

    for (const auto& [u, weight] : graph.adjacency()[v]) {

        if (partition_[u] == partition_[v])
            delta_[u] += 2.0 * weight;
        else
            delta_[u] -= 2.0 * weight;
    }
}

void CutState::set_partition(const std::vector<int8_t>& p) {
    partition_ = p;
    calculate();
}

double CutState::cut_weight() const {
    return cut_weight_;
}

int CutState::cut_edge_count() const {
    return cut_edge_count_;
}

const std::vector<int8_t>& CutState::partition() const {
    return partition_;
}

const std::vector<double>& CutState::delta() const {
    return delta_;
}

double CutState::flip_delta(int v) const {
    return delta_[v];
}