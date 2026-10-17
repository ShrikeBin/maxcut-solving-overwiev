#include "Graph.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

Graph::Graph(const std::string& filename) {
    load(filename);
}

void Graph::load(const std::string& filename) {

    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open graph file: " + filename
        );
    }

    std::string line;

    // Directed
    std::getline(file, line);
    isDirected = std::stoi(line.substr(line.find('=') + 1)) != 0;

    // Weighted
    std::getline(file, line);
    isWeighted = std::stoi(line.substr(line.find('=') + 1)) != 0;

    // Vertex count
    std::getline(file, line);
    V_count = std::stoi(line.substr(line.find('=') + 1));

    // Edge count
    std::getline(file, line);
    E_count = std::stoi(line.substr(line.find('=') + 1));

    adj.resize(V_count);

    while (std::getline(file, line)) {

        // Trim whitespace
        if (line.empty())
            continue;

        // Comments
        if (line.rfind("/*", 0) == 0 ||
            line.rfind("*/", 0) == 0)
            continue;
        std::stringstream ss(line);

        std::string u_string;
        std::string v_string;
        std::string weight_string;

        std::getline(ss, u_string, ';');
        std::getline(ss, v_string, ';');

        if (u_string.empty() || v_string.empty())
            continue;

        int u = std::stoi(u_string);
        int v = std::stoi(v_string);

        double weight = 1.0;

        if (isWeighted) {
            std::getline(ss, weight_string, ';');
            weight = std::stod(weight_string);
        }

        // Your files appear to use 1-based vertices.
        // C++ internally uses 0-based.
        --u;
        --v;

        // Keep graph undirected.
        edge_list.push_back({u, v, weight});

        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight});
    }
}

int Graph::vertex_count() const {
    return V_count;
}

int Graph::edge_count() const {
    return E_count;
}

const std::vector<Edge>& Graph::edges() const {
    return edge_list;
}

const std::vector<std::vector<std::pair<int, double>>>&
Graph::adjacency() const {
    return adj;
}

double Graph::edge_weight(int u, int v) const {

    for (const auto& [neighbor, weight] : adj[u]) {
        if (neighbor == v)
            return weight;
    }

    return 0.0;
}