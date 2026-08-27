import numpy as np
import cvxpy as cp
import time
from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result

import mosek.fusion as mf
import sys

def goemans_williamson(graph: Graph, RANDOM_SLICE_TRIALS: int, SOLVE_VERBOSE: bool) -> Result:
    """
    The Goemans-Williamson 0.878-approximation algorithm for Max-Cut.
    Relaxation into a Semidefinite Program (SDP), followed by random hyperplane rounding.
    Direct MOSEK API call for better RAM usage
    """
    start_time = time.perf_counter()
    n = graph.V_count
    
    # Build Weight Matrix
    W = np.zeros((n, n))
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            weight = graph.matrix[u][v]
            if weight != 0.0:
                W[u - 1][v - 1] = weight
                W[v - 1][u - 1] = weight
    
    total_W = np.sum(W)

    # Initialize MOSEK Model
    with mf.Model("MaxCut_SDP") as M:
        if SOLVE_VERBOSE:
            M.setLogHandler(sys.stdout)
            
        # Define X as a Symmetric Positive Semidefinite Matrix Variable
        X = M.variable("X", mf.Domain.inPSDCone(n))
        
        # Diagonal constraint: X_ii == 1
        M.constraint("diag_one", X.diag(), mf.Domain.equalsTo(1.0))
        
        # Maximize: 0.25 * (total_W - Tr(W * X))
        # Expr.dot performs an efficient elementwise inner product
        M.objective(
            mf.ObjectiveSense.Maximize, 
            mf.Expr.mul(0.25, mf.Expr.sub(total_W, mf.Expr.dot(W, X)))
        )
        
        # Solve
        M.solve()
        
        # Extract Solution Matrix
        X_val = np.array(X.level()).reshape((n, n))

    # Cholesky & Hyperplane Rounding
    eigenvalues, eigenvectors = np.linalg.eigh(X_val)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    X_projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    V = np.linalg.cholesky(X_projected)
    
    R = np.random.normal(0, 1, (n, RANDOM_SLICE_TRIALS))
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    S = (V @ R) >= 0
    
    best_partition = {}
    best_weight = -1.0
    best_edge_count = 0

    for t in range(RANDOM_SLICE_TRIALS):
        current_partition = {i + 1: int(S[i, t]) for i in range(n)}
        weight, edge_count = graph.get_cut_info(current_partition)
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

def _goemans_williamson_OLD(graph: Graph, RANDOM_SLICE_TRIALS: int, SOLVE_VERBOSE: bool) -> Result:
    """
    The Goemans-Williamson 0.878-approximation algorithm for Max-Cut.
    Relaxation into a Semidefinite Program (SDP), followed by random hyperplane rounding.
    """
    start_time = time.perf_counter()
    n = graph.V_count
    
    # 1. Build a fully symmetric Weight matrix from the upper-triangle graph structure (indexxed from 0)
    W = np.zeros((n, n))
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            weight = graph.matrix[u][v]
            if weight != 0.0:
                W[u - 1][v - 1] = weight
                W[v - 1][u - 1] = weight
        
    # 2. Define and solve the SDP relaxation using CVXPY
    X = cp.Variable((n, n), PSD=True)
    
    # Max-Cut objective formula: 0.25 * sum(W_ij * (1 - X_ij))
    objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - X)))
    constraints = [cp.diag(X) == 1]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=SOLVE_VERBOSE)
    
    if X.value is None:
        raise ValueError("SDP optimization failed to converge.")
        
    eigenvalues, eigenvectors = np.linalg.eigh(X.value)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    X_projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    V = np.linalg.cholesky(X_projected)
    
    best_partition = {}
    best_weight = -1.0
    best_edge_count = 0

    # Slice the optimized sphere N times
    for _ in range(RANDOM_SLICE_TRIALS):
        r = np.random.normal(0, 1, n)
        r_norm = np.linalg.norm(r)
        if r_norm > 0:
            r /= r_norm
        
        current_partition = {}
        for i in range(n):
            sign = np.dot(V[i], r)
            current_partition[i + 1] = 1 if sign >= 0 else 0
            
        weight, edge_count = graph.get_cut_info(current_partition)
        
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