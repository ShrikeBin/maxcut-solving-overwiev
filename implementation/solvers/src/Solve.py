from mip import Model, xsum, MAXIMIZE, BINARY, GRB
import time
from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result
from collections import defaultdict








# FIX DIRECTIONAL BUG




















def solve_mip(graph: Graph, SOLVE_VERBOSE: bool) -> Result:
    """
    Solves Max-Cut problem using MIP model
    """
    start_time = time.perf_counter()
    model = Model("Max-Cut", sense=MAXIMIZE, solver_name=GRB)
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
    
    if not model.num_solutions:
        return Result(partition={}, cut_weight=0.0)
        
    partition = {k: 1 if v[k].x >= 0.5 else 0 for k in v}
    
    max_cut_weight = model.objective_value
    _, edge_count = graph.get_cut_info(partition)
    elapsed_time = time.perf_counter() - start_time
    
    return Result(
        partition=partition, 
        cut_weight=max_cut_weight,
        cut_edge_count=edge_count, 
        type="Exact MIP",
        execution_time=elapsed_time
    )

# def solve_mip(graph: Graph, SOLVE_VERBOSE: bool) -> Result:
#     """
#     Solves Max-Cut problem using MIP model safely without double-counting edge weights.
#     """
#     start_time = time.perf_counter()
#     model = Model("Max-Cut", sense=MAXIMIZE, solver_name=GRB)
#     model.verbose = 1 if SOLVE_VERBOSE else 0
    
#     v = {k: model.add_var(var_type=BINARY, name=f"v_{k}") for k in range(1, graph.V_count + 1)}
#     combined_weights = defaultdict(float)

#     if graph.useAdjList:
#         for u in graph.adj_list:
#             for neighbor, weight in graph.adj_list[u]:
#                 if u == neighbor:
#                     continue 
#                 # FIX: For undirected graphs, only look at each pair once
#                 if not graph.isDirected and u > neighbor:
#                     continue
                
#                 if graph.isDirected:
#                     # For directed graphs, treat (u, v) and (v, u) uniquely if needed,
#                     # or combine them depending on your Max-Cut definition.
#                     pair = (u, neighbor)
#                 else:
#                     pair = (min(u, neighbor), max(u, neighbor))
                    
#                 combined_weights[pair] += weight
#     else:
#         for u in range(1, graph.V_count + 1):
#             # u + 1 ensures neighbor > u, so we only scan the upper triangle of the matrix
#             for neighbor in range(u + 1, graph.V_count + 1):
#                 if graph.isDirected:
#                     w1 = graph.matrix[u][neighbor]
#                     if w1 != 0.0:
#                         combined_weights[(u, neighbor)] = w1
#                     w2 = graph.matrix[neighbor][u]
#                     if w2 != 0.0:
#                         combined_weights[(neighbor, u)] = w2
#                 else:
#                     # FIX: Read the edge once. If it's an undirected matrix, 
#                     # graph.matrix[u][neighbor] holds the true edge weight.
#                     w = graph.matrix[u][neighbor] or graph.matrix[neighbor][u]
#                     if w != 0.0:
#                         combined_weights[(u, neighbor)] = w

#     edges = set()
#     e = {}
    
#     for (u, neighbor), weight in combined_weights.items():
#         edges.add((u, neighbor, weight))
#         edge_var = model.add_var(var_type=BINARY, name=f"e_{u}_{neighbor}")
#         e[(u, neighbor)] = edge_var
        
#         # Upper bounds: Force edge_var to 0 if nodes are on the same side
#         model += edge_var <= v[u] + v[neighbor], f"cut_ub1_{u}_{neighbor}"
#         model += edge_var <= 2 - (v[u] + v[neighbor]), f"cut_ub2_{u}_{neighbor}"
        
#         # FIX / OPTIMIZATION: Lower bounds enforce standard linearization |v_u - v_v|
#         # This keeps the linear relaxation tight so Gurobi can solve it much faster!
#         model += edge_var >= v[u] - v[neighbor], f"cut_lb1_{u}_{neighbor}"
#         model += edge_var >= v[neighbor] - v[u], f"cut_lb2_{u}_{neighbor}"

#     model.objective = xsum(weight * e[(u, neighbor)] for u, neighbor, weight in edges)
        
#     model.optimize()
    
#     # Extract the resulting partition dictionary mapping node -> side
#     # (Using int(var.x) or int(var.value) depending on python-mip vs native gurobipy syntax)
#     partition = {k: int(v[k].value if hasattr(v[k], 'value') else v[k].x) for k in v}
    
#     final_weight, edge_count = graph.get_cut_info(partition)
#     elapsed_time = time.perf_counter() - start_time
    
#     return Result(
#         partition=partition, 
#         cut_weight=final_weight,
#         cut_edge_count=edge_count, 
#         type="Exact MIP",
#         execution_time=elapsed_time
#     )