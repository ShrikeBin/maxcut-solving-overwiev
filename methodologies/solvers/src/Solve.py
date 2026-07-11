from mip import Model, xsum, MAXIMIZE, BINARY
import sys
from src.Graph import Graph
from collections import defaultdict

def solve_mip(graph: Graph, PRINT_RESULTS: bool, PRINT_EDGES: bool, SOLVE_VERBOSE: bool):
    model = Model("Max-Cut", sense=MAXIMIZE)
    model.verbose = 1 if SOLVE_VERBOSE else 0
    v = {k: model.add_var(var_type=BINARY, name=f"v_{k}") for k in range(1, graph.V_count + 1)}
    
    combined_weights = defaultdict(float)

    if graph.useAdjList:
        for u in graph.adj_list:
            for neighbor, weight in graph.adj_list[u]:
                if u == neighbor:
                    continue 
                pair = (min(u, neighbor), max(u, neighbor))
                combined_weights[pair] += weight
    else:
        for u in range(1, graph.V_count + 1):
            for neighbor in range(u + 1, graph.V_count + 1):
                w1 = graph.matrix[u][neighbor]
                w2 = graph.matrix[neighbor][u]
                if w1 != 0.0 or w2 != 0.0:
                    combined_weights[(u, neighbor)] = w1 + w2

    edges = set()
    e = {}
    
    for (u, neighbor), weight in combined_weights.items():
        edges.add((u, neighbor, weight))
        edge_var = model.add_var(var_type=BINARY, name=f"e_{u}_{neighbor}")
        e[(u, neighbor)] = edge_var
        
        model += edge_var <= v[u] + v[neighbor], f"cut_ub1_{u}_{neighbor}"
        model += edge_var <= 2 - (v[u] + v[neighbor]), f"cut_ub2_{u}_{neighbor}"

    model.objective = xsum(weight * e[(u, neighbor)] for u, neighbor, weight in edges)
        
    model.optimize()
    
    if PRINT_RESULTS:
        print("\n--- Optimization Results ---")
        if model.num_solutions:
            print(f"Status: Optimal Solution Found")
            print(f"Maximum Cut Weight: {model.objective_value}")
            
            set_A = [k for k in v if v[k].x >= 0.5]
            set_B = [k for k in v if v[k].x < 0.5]
            print(f"Partition A (v=1): {set_A}")
            print(f"Partition B (v=0): {set_B}")
            
            if PRINT_EDGES:
                print("\nEdges in the Cut:")
                for u, neighbor, weight in edges:
                    if e[(u, neighbor)].x >= 0.5:
                        print(f"  Edge ({u} - {neighbor}) with combined weight {weight}")
        else:
            print("No solution found.")