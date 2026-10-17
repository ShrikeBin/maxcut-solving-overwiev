#ifndef GRAPH_HPP
#define GRAPH_HPP

#pragma once

#include <vector>
#include <string>
#include <utility>

struct Edge {
    int u;
    int v;
    double weight;
};

class Graph {
public:
    Graph(const std::string& filename);

    int vertex_count() const;
    int edge_count() const;

    const std::vector<Edge>& edges() const;
    const std::vector<std::vector<std::pair<int, double>>>& adjacency() const;

    double edge_weight(int u, int v) const;

private:
    int V_count = 0;
    int E_count = 0;

    bool isDirected = false;
    bool isWeighted = false;

    std::vector<Edge> edge_list;

    // adjacency[v] = {(neighbor, weight), ...}
    std::vector<std::vector<std::pair<int, double>>> adj;

    void load(const std::string& filename);
};

#endif // GRAPH_HPP