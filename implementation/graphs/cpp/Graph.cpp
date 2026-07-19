#include "Graph.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// private for file namespace
namespace {
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    int parseHeaderValue(const std::string& line) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            return std::stoi(trim(line.substr(pos + 1)));
        }
        return 0;
    }
}

Graph::Graph(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    
    if (std::getline(file, line)) isDirected = parseHeaderValue(line);
    if (std::getline(file, line)) isWeighted = parseHeaderValue(line);
    if (std::getline(file, line)) V_count = parseHeaderValue(line);
    if (std::getline(file, line)) E_count = parseHeaderValue(line);


    matrix.assign(V_count + 1, std::vector<double>(V_count + 1, 0.0));
    

    // Parse data rows
    while (std::getline(file, line)) {
        std::string clean_line = trim(line);
        if (clean_line.empty() || clean_line.rfind("/*", 0) == 0 || clean_line.rfind("*/", 0) == 0) {
            continue;
        }

        std::stringstream ss(clean_line);
        std::string part;
        std::vector<std::string> parts;
        
        while (std::getline(ss, part, ';')) {
            parts.push_back(trim(part));
        }

        if (parts.size() >= 2) {
            int u = std::stoi(parts[0]);
            int v = std::stoi(parts[1]);
            double weight = (isWeighted && parts.size() >= 3) ? std::stod(parts[2]) : 1.0;

  
            matrix[u][v] = weight;
            if (!isDirected) {
                matrix[v][u] = weight;
            }
            
        }
    }
}

void Graph::print() const {
    
    std::cout << "--- Graph Properties ---\n";
    std::cout << "Directed  = " << isDirected << "\n";
    std::cout << "Weighted  = " << isWeighted << "\n";
    std::cout << "     |V|  = " << V_count << "\n";
    std::cout << "     |E|  = " << E_count << "\n";
    std::cout << "------------------------\n";


    std::cout << "    ";
    for (int i = 1; i <= V_count; ++i) {
        std::cout << std::setw(2) << i << "  ";
    }
    std::cout << "\n";

    for (int i = 1; i <= V_count; ++i) {
        std::cout << std::setw(2) << i << ": ";
        for (int j = 1; j <= V_count; ++j) {
            double val = matrix[i][j];
            if (val == 0.0) {
                std::cout << " 0";
            } else {
                if (val == static_cast<int>(val)) {
                    std::cout << std::setw(2) << static_cast<int>(val);
                } else {
                    std::cout << std::fixed << std::setprecision(1) << val;
                }
            }
            if (j < V_count) std::cout << "  ";
        }
        std::cout << "\n";
    }
    
    std::cout << "------------------------\n";
}