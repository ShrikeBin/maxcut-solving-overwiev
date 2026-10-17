from mip import Model, xsum, MAXIMIZE, BINARY, GRB
import time
from implementation.util.python.Graph import Graph
from implementation.util.python.Result import Result

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
    it, either on every incumbent/bound change (gap_checkpoints=None) or
    specifically when the relative MIP gap crosses each threshold in
    gap_checkpoints (e.g. [0.15, 0.12, 0.09, 0.06, 0.03, 0.01]).

    Also reports the best incumbent as a final checkpoint if the time limit hits.

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

    def extract_result(var_values: dict, tag: str) -> Result:
        partition = {k: (1 if var_values[k] >= 0.5 else 0) for k in v}
        # Recompute from the partition itself, not the model objective,
        # since the e-vars are only upper-bounded and can undercount.
        true_weight, edge_count = graph.get_cut_info(partition)
        elapsed = time.perf_counter() - start_time
        return Result(
            partition=partition,
            cut_weight=true_weight,
            cut_edge_count=edge_count,
            type=f"Exact MIP (Gurobi native, {tag})",
            execution_time=elapsed,
        )

    # Best incumbent (true cut weight) and last seen bound, shared by the callback
    state = {"best_obj": None, "best_vals": None, "last_bound": None}

    def report_if_needed(bound):
        nonlocal remaining_checkpoints
        if state["best_vals"] is None or abs(bound) >= gp.GRB.INFINITY:
            return

        best = state["best_obj"]
        gap = abs(bound - best) / max(1e-10, abs(best))

        if not checkpoints:
            should_report = True
        else:
            should_report = False
            while remaining_checkpoints and gap <= remaining_checkpoints[0]:
                should_report = True
                remaining_checkpoints.pop(0)

        if should_report and on_checkpoint:
            result = extract_result(state["best_vals"], f"gap<={gap*100:.1f}%")
            on_checkpoint(result, gap)

    def callback(m, where):
        if where == gp.GRB.Callback.MIPSOL:
            # New incumbent (cbGetSolution only works in this branch)
            vals = {k: m.cbGetSolution(var) for k, var in v.items()}
            partition = {k: (1 if vals[k] >= 0.5 else 0) for k in v}
            true_weight, _ = graph.get_cut_info(partition)

            if state["best_obj"] is None or true_weight > state["best_obj"]:
                state["best_obj"] = true_weight
                state["best_vals"] = vals

            bound = m.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
            state["last_bound"] = bound
            report_if_needed(bound)

        elif where == gp.GRB.Callback.MIP:
            # Periodic call: catches BestBd changes with no new incumbent
            if state["best_vals"] is None:
                return

            bound = m.cbGet(gp.GRB.Callback.MIP_OBJBND)
            last = state["last_bound"]
            if last is not None and abs(bound - last) < 1e-6:
                return  # bound unchanged, nothing new to report

            state["last_bound"] = bound
            report_if_needed(bound)

    model.optimize(callback)

    if model.SolCount == 0:
        return Result(
            partition={}, cut_weight=0.0, cut_edge_count=0,
            type="Exact MIP (Gurobi native, no solution found)",
            execution_time=time.perf_counter() - start_time,
        )

    hit_time_limit = (model.Status == gp.GRB.TIME_LIMIT)

    # Time ran out: report the best incumbent as one last checkpoint
    if hit_time_limit and on_checkpoint:
        var_values = {k: v[k].X for k in v}
        final_gap = model.MIPGap  # gap between best incumbent and best bound at termination
        result = extract_result(var_values, f"time limit, gap={final_gap*100:.1f}%")
        on_checkpoint(result, final_gap)

    partition = {k: (1 if v[k].X >= 0.5 else 0) for k in v}
    max_cut_weight, edge_count = graph.get_cut_info(partition)
    elapsed_time = time.perf_counter() - start_time

    return Result(
        partition=partition,
        cut_weight=max_cut_weight,
        cut_edge_count=edge_count,
        type="Exact MIP (Gurobi native, time limit)" if hit_time_limit else "Exact MIP (Gurobi native)",
        execution_time=elapsed_time,
    )