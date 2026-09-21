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


def solve_with_partial_results(
    graph: Graph,
    SOLVE_VERBOSE: bool,
    time_limit: float = None,
    gap_checkpoints: list = None,
    on_checkpoint=None,
) -> Result:
    """
    Abstraction layer over solve(). When gurobipy is available, builds and solves
    the Max-Cut model natively with Gurobi's Python API to get true incumbent
    callbacks -- the best-known solution is reported live as the solver improves
    it, either on every improving solution (gap_checkpoints=None) or specifically
    when the relative MIP gap crosses each threshold in gap_checkpoints
    (e.g. [0.05, 0.01, 0.001]).

    Returns None if gurobipy isn't installed.
    """
    try:
        import gurobipy as gp
    except ImportError:
        print("gurobipy not available -- Partial results only supported with Gurobi - returning <None>")
        return None

    start_time = time.perf_counter()
    checkpoints = sorted(gap_checkpoints, reverse=True) if gap_checkpoints else []
    remaining_checkpoints = list(checkpoints)

    model = gp.Model("Max-Cut")
    model.Params.OutputFlag = 1 if SOLVE_VERBOSE else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    v = {k: model.addVar(vtype=gp.GRB.BINARY, name=f"v_{k}") for k in range(1, graph.V_count + 1)}
    e = {}
    edges_to_score = []
    for u in range(1, graph.V_count + 1):
        for neighbor in range(u + 1, graph.V_count + 1):
            weight = graph.matrix[u][neighbor]
            if weight != 0.0:
                edges_to_score.append((u, neighbor, weight))
                edge_var = model.addVar(vtype=gp.GRB.BINARY, name=f"e_{u}_{neighbor}")
                e[(u, neighbor)] = edge_var
                model.addConstr(edge_var <= v[u] + v[neighbor], name=f"cut_ub1_{u}_{neighbor}")
                model.addConstr(edge_var <= 2 - (v[u] + v[neighbor]), name=f"cut_ub2_{u}_{neighbor}")

    model.setObjective(
        gp.quicksum(weight * e[(u, neighbor)] for u, neighbor, weight in edges_to_score),
        gp.GRB.MAXIMIZE,
    )
    model.update()

    def extract_result(var_values: dict, obj_val: float, tag: str) -> Result:
        partition = {k: (1 if var_values[k] >= 0.5 else 0) for k in v}
        _, edge_count = graph.get_cut_info(partition)
        elapsed = time.perf_counter() - start_time
        return Result(
            partition=partition,
            cut_weight=obj_val,
            cut_edge_count=edge_count,
            type=f"Exact MIP (Gurobi native, {tag})",
            execution_time=elapsed,
        )

    def callback(m, where):
        nonlocal remaining_checkpoints
        if where == gp.GRB.Callback.MIPSOL:
            obj_best = m.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)
            obj_bound = m.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
            gap = abs(obj_best - obj_bound) / max(1e-10, abs(obj_best))

            if not checkpoints:
                should_report = True
            else:
                should_report = False
                while remaining_checkpoints and gap <= remaining_checkpoints[0]:
                    should_report = True
                    remaining_checkpoints.pop(0)

            if should_report and on_checkpoint:
                var_values = {k: m.cbGetSolution(var) for k, var in v.items()}
                result = extract_result(var_values, obj_best, f"gap<={gap:.4f}")
                on_checkpoint(result, gap)

    model.optimize(callback)

    if model.SolCount == 0:
        return Result(
            partition={}, cut_weight=0.0, cut_edge_count=0,
            type="Exact MIP (Gurobi native)", execution_time=0.0,
        )

    partition = {k: (1 if v[k].X >= 0.5 else 0) for k in v}
    max_cut_weight = model.ObjVal
    _, edge_count = graph.get_cut_info(partition)
    elapsed_time = time.perf_counter() - start_time
    return Result(
        partition=partition,
        cut_weight=max_cut_weight,
        cut_edge_count=edge_count,
        type="Exact MIP (Gurobi native)",
        execution_time=elapsed_time,
    )