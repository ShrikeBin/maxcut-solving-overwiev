from mip import Model, xsum, MAXIMIZE, BINARY, GRB
import time
from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result

def solve(graph: Graph, SOLVE_VERBOSE: bool) -> Result:
    """
    Solves Max-Cut problem using an exact MIP model.
    """
    start_time = time.perf_counter()
    model = Model("Max-Cut", sense=MAXIMIZE, solver_name=GRB)
    model.verbose = 1 if SOLVE_VERBOSE else 0
    
    v = {k: model.add_var(var_type=BINARY, name=f"v_{k}") for k in range(1, graph.V_count + 1)}
    
    edges_to_score = []
    e = {}
    
    # Strictly iterate through the upper triangle where the total collapsed weights are
    for u in range(1, graph.V_count + 1):
        for neighbor in range(u + 1, graph.V_count + 1):
            weight = graph.matrix[u][neighbor]
            if weight != 0.0:
                edges_to_score.append((u, neighbor, weight))
                
                edge_var = model.add_var(var_type=BINARY, name=f"e_{u}_{neighbor}")
                e[(u, neighbor)] = edge_var
                
                model += edge_var <= v[u] + v[neighbor], f"cut_ub1_{u}_{neighbor}"
                model += edge_var <= 2 - (v[u] + v[neighbor]), f"cut_ub2_{u}_{neighbor}"

    model.objective = xsum(weight * e[(u, neighbor)] for u, neighbor, weight in edges_to_score)
        
    model.optimize()
    
    if not model.num_solutions:
        return Result(partition={}, cut_weight=0.0, cut_edge_count=0, type="Exact MIP", execution_time=0.0)
        
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