#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <string>
#include <vector>
#include <map>
#include <utility>

class Graph {
private:
    bool useAdjList;
    int isDirected;
    int isWeighted;
    int V_count;
    int E_count;

    std::map<u_int, std::vector<std::pair<u_int, double>>> adj_list;
    
    std::vector<std::vector<double>> matrix;

public:
    Graph(const std::string& filename, bool useAdjList = true);

    void print() const;
};

#endif // GRAPH_HPP