from mip import Model, xsum, MAXIMIZE, BINARY
import sys
from src.Graph import Graph

def solve_mip(graph: Graph, PRINT_RESULTS: bool, PRINT_EDGES: bool, SOLVE_VERBOSE: bool):
    model = Model("Max-Cut", sense=MAXIMIZE)
    model.verbose = 1 if SOLVE_VERBOSE else 0
    v = {k: model.add_var(var_type=BINARY, name=f"v_{k}") for k in range(1, graph.V_count + 1)}
    edges = set()
    e = {}
    
    # Load graph data and initialize variables
    if graph.useAdjList:
        for u in graph.adj_list:
            for neighbor, weight in graph.adj_list[u]:
                # For undirected - ensure each edge is added once
                if not graph.isDirected and u > neighbor:
                    continue
                edge = (u, neighbor)
                edges.add((u, neighbor, weight))
                e[edge] = model.add_var(var_type=BINARY, name=f"e_{u}_{neighbor}")
    else:
        for u in range(1, graph.V_count + 1):
            for neighbor in range(1, graph.V_count + 1):
                weight = graph.matrix[u][neighbor]
                if weight != 0.0:
                    if not graph.isDirected and u > neighbor:
                        continue
                    edge = (u, neighbor)
                    edges.add((u, neighbor, weight))
                    e[edge] = model.add_var(var_type=BINARY, name=f"e_{u}_{neighbor}")

    # Objective function
    model.objective = xsum(weight * e[(u, neighbor)] for u, neighbor, weight in edges)
    
    # Constraints for every edge:
    # e_ij <= v_i + v_j
    # e_ij <= 2 - (v_i + v_j)
    for u, neighbor, _ in edges:
        model += e[(u, neighbor)] <= v[u] + v[neighbor], f"cut_ub1_{u}_{neighbor}"
        model += e[(u, neighbor)] <= 2 - (v[u] + v[neighbor]), f"cut_ub2_{u}_{neighbor}"
        
    # Run model
    model.optimize()
    
    # Results
    if(PRINT_RESULTS):
        print("\n--- Optimization Results ---")
        if model.num_solutions:
            print(f"Status: Optimal Solution Found")
            print(f"Maximum Cut Weight: {model.objective_value}")
            
            # Display vertex partitions
            set_A = [k for k in v if v[k].x >= 0.5]
            set_B = [k for k in v if v[k].x < 0.5]
            print(f"Partition A (v=1): {set_A}")
            print(f"Partition B (v=0): {set_B}")
            
            if(PRINT_EDGES):
                print("\nEdges in the Cut:")
                for u, neighbor, weight in edges:
                    if e[(u, neighbor)].x >= 0.5:
                        print(f"  Edge ({u} - {neighbor}) with weight {weight}")
        else:
            print("No solution found.")

    
