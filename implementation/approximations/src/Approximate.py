import numpy as np
import cvxpy as cp
import time
from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result

def goemans_williamson(graph: Graph, RANDOM_SLICE_TRIALS: int, SOLVE_VERBOSE: bool) -> Result:
    """
    The Goemans-Williamson 0.878-approximation algorithm for Max-Cut.
    Relaxation into a Semidefinite Program (SDP), followed by random hyperplane rounding.
    """
    start_time = time.perf_counter()
    n = graph.V_count
    
    # 1. Build the Weight matrix from the graph structure (1-based to 0-based index shift)
    W = np.zeros((n, n))
    if graph.useAdjList:
        for u, neighbors in graph.adj_list.items():
            for v, weight in neighbors:
                W[u - 1][v - 1] = weight
    else:
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                W[u - 1][v - 1] = graph.matrix[u][v]
                
    # Balance symmetrically for undirected structures
    if not graph.isDirected:
        W = (W + W.T) / 2.0
        
    # 2. Define and solve the SDP relaxation using CVXPY
    X = cp.Variable((n, n), PSD=True)
    
    # Max-Cut objective formula: 0.25 * sum(W_ij * (1 - X_ij))
    objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - X)))
    constraints = [cp.diag(X) == 1]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=SOLVE_VERBOSE)  # SCS is an efficient solver bundled natively with CVXPY
    
    if X.value is None:
        raise ValueError("SDP optimization failed to converge.")
        
    # 3. Securely factorize X by projecting it onto the PSD cone
    # This completely eliminates Cholesky precision errors from solvers like SCS
    eigenvalues, eigenvectors = np.linalg.eigh(X.value)
    
    # Clip all eigenvalues to be strictly positive (at least 1e-9)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    
    # Reconstruct the clean, perfectly positive definite matrix
    X_projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # This will now pass safely without ever crashing!
    V = np.linalg.cholesky(X_projected)
    
    best_partition = {}
    best_weight = -1.0
    best_edge_count = 0

    # Slice the optimized sphere N times
    for _ in range(RANDOM_SLICE_TRIALS):
        # 4. Generate a new random hyperplane
        r = np.random.normal(0, 1, n)
        r_norm = np.linalg.norm(r)
        if r_norm > 0:
            r /= r_norm
        
        # 5. Partition based on this specific slice
        current_partition = {}
        for i in range(n):
            sign = np.dot(V[i], r)
            current_partition[i + 1] = 1 if sign >= 0 else 0
            
        # Evaluate how good this specific random slice was
        weight, edge_count = graph.get_cut_info(current_partition)
        
        # Save it if it beats our previous random slices
        if weight > best_weight:
            best_weight = weight
            best_edge_count = edge_count
            best_partition = current_partition

    elapsed_time = time.perf_counter() - start_time
    
    return Result(
        partition=best_partition, 
        cut_weight=best_weight,
        cut_edge_count=best_edge_count,
        type=f"Goemans-Williamson ({RANDOM_SLICE_TRIALS} Slices)",
        execution_time=elapsed_time
    )


def randomized_half(graph: Graph, TRIALS: int) -> Result:
    """
    A randomized 0.5-approximation algorithm for the Max-Cut problem.
    Runs for TRIALS iterations, assigning vertices to a side uniformly at random 
    each time, returning the best configuration found.
    """
    start_time = time.perf_counter()
    
    best_cut_partition = None
    best_final_weight = -float('inf')
    best_edge_count = 0
    
    for _ in range(TRIALS):
        cut_partition = {v: int(np.random.choice([0, 1])) for v in range(1, graph.V_count + 1)}
        cut_weight, edge_count = graph.get_cut_info(cut_partition)        
        
        if cut_weight > best_final_weight:
            best_final_weight = cut_weight
            best_edge_count = edge_count
            best_cut_partition = cut_partition.copy()
            
    elapsed_time = time.perf_counter() - start_time
    
    return Result(
        partition=best_cut_partition, 
        cut_weight=best_final_weight,
        cut_edge_count=best_edge_count,
        type=f"Randomized 0.5 (Best out of {TRIALS})",
        execution_time=elapsed_time
    )

def randomized_greedy_edges(graph: Graph, TRIALS: int) -> Result:
    """
    One-pass randomized greedy algorithm browsing edges in a random order.
    Runs for TRIALS iterations, tracking and returning the best overall cut.
    """
    start_time = time.perf_counter()
    n = graph.V_count
    
    # 1. Collect all unique edges in the graph ONCE to maximize speed
    base_edges = []
    if graph.useAdjList:
        for u, neighbors in graph.adj_list.items():
            for v, weight in neighbors:
                if not graph.isDirected and u > v:
                    continue
                base_edges.append((u, v, weight))
    else:
        for u in range(1, n + 1):
            start_v = 1 if graph.isDirected else u + 1
            for v in range(start_v, n + 1):
                weight = graph.matrix[u][v]
                if weight != 0.0:
                    base_edges.append((u, v, weight))
                    
    best_cut_partition = None
    best_final_weight = -float('inf')
    best_edge_count = 0

    # 2. Run the multi-trial loop
    for _ in range(TRIALS):
        cut_partition = {v: -1 for v in range(1, n + 1)}
        
        # Make a copy of the edge list and shuffle its order for this trial
        edges = base_edges.copy()
        np.random.shuffle(edges)
        
        # Process the shuffled edges
        for u, v, weight in edges:
            p_u = cut_partition[u]
            p_v = cut_partition[v]
            
            if p_u == -1 and p_v == -1:
                side = int(np.random.choice([0, 1]))
                cut_partition[u] = side
                cut_partition[v] = 1 - side  
                
            elif p_u != -1 and p_v == -1:
                cut_partition[v] = 1 - p_u   
                
            elif p_u == -1 and p_v != -1:
                cut_partition[u] = 1 - p_v   
                
            else:
                continue

        # Clean up isolated nodes
        for v in cut_partition:
            if cut_partition[v] == -1:
                cut_partition[v] = int(np.random.choice([0, 1]))
                
        # Evaluate this trial
        final_weight, edge_count = graph.get_cut_info(cut_partition)
        
        if final_weight > best_final_weight:
            best_final_weight = final_weight
            best_edge_count = edge_count
            best_cut_partition = cut_partition.copy()
            
    elapsed_time = time.perf_counter() - start_time
    
    return Result(
        partition=best_cut_partition, 
        cut_weight=best_final_weight,
        cut_edge_count=best_edge_count, 
        type=f"Randomized Greedy Edges (Best out of {TRIALS})",
        execution_time=elapsed_time
    )

def randomized_greedy_vertices(graph: Graph, TRIALS: int) -> Result:
    """
    Runs the randomized greedy vertex algorithm multiple times (TRIALS),
    tracking and returning the best overall cut found.
    """
    start_time = time.perf_counter()
    n = graph.V_count
    
    best_cut_partition = None
    best_final_weight = -float('inf')
    best_edge_count = 0

    for _ in range(TRIALS):
        # 1. Start with an initial random assignment for this trial
        cut_partition = {v: int(np.random.choice([0, 1])) for v in range(1, n + 1)}
        
        # 2. Get a randomized sequence of vertices
        nodes = list(range(1, n + 1))
        np.random.shuffle(nodes)
        
        # 3. Process every vertex exactly once
        for u in nodes:
            current_partition = cut_partition[u]
            opposite_partition = 1 - current_partition
            
            weight_internal = 0.0
            weight_crossing = 0.0
            
            if graph.useAdjList:
                neighbors = graph.adj_list[u]
            else:
                neighbors = []
                for v in range(1, n + 1):
                    # Fixed potential index lookup pattern safely
                    w = graph.matrix[u][v] if graph.isDirected else (graph.matrix[u][v] or graph.matrix[v][u])
                    if w != 0.0:
                        neighbors.append((v, w))
            
            for v, weight in neighbors:
                if cut_partition[v] == current_partition:
                    weight_internal += weight
                else:
                    weight_crossing += weight
            
            # Make a one-time greedy correction if the node is on the wrong side
            if weight_internal > weight_crossing:
                cut_partition[u] = opposite_partition
                
        # Calculate cut value for this specific trial
        final_weight, edge_count = graph.get_cut_info(cut_partition)
        
        # Track the absolute best results across all trials
        if final_weight > best_final_weight:
            best_final_weight = final_weight
            best_edge_count = edge_count
            best_cut_partition = cut_partition.copy()
            
    elapsed_time = time.perf_counter() - start_time
    
    return Result(
        partition=best_cut_partition, 
        cut_weight=best_final_weight, 
        cut_edge_count=best_edge_count,
        type=f"Randomized Greedy Vertices (Best out of {TRIALS})",
        execution_time=elapsed_time
    )