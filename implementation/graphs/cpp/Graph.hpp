#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <string>
#include <vector>
#include <map>
#include <utility>

class Graph {
private:
    int isDirected;
    int isWeighted;
    int V_count;
    int E_count;

    std::vector<std::vector<double>> matrix;

public:
    Graph(const std::string& filename);

    void print() const;
};

#endif // GRAPH_HPP