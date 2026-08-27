import ctypes
import os
import time
from typing import Optional
from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result

# 1. Resolve path to shared library compiled by the Makefile
_lib_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "../cpp/lib/libheuristics_memetics.so"
    )
)

if not os.path.exists(_lib_path):
    raise FileNotFoundError(
        f"Could not find shared C++ library at: {_lib_path}\n"
        "Please run 'make' inside 'implementation/methodologies/heuristics_memetics/cpp/' first."
    )

_lib = ctypes.CDLL(_lib_path)


# 2. Define standard C Return Struct matching Bind.cpp
class _CResult(ctypes.Structure):
    _fields_ = [
        ("cut_weight", ctypes.c_double),
        ("cut_edge_count", ctypes.c_int),
        ("partition", ctypes.POINTER(ctypes.c_int))
    ]


# 3. Configure Function Signatures

# Memory cleanup
_lib.free_c_result.argtypes = [_CResult]

# Heuristics
_lib.run_local_search.argtypes = [
    ctypes.c_char_p,              # Graph filename
    ctypes.c_uint64,              # Seed
    ctypes.POINTER(ctypes.c_int8) # Initial partition pointer (or NULL)
]
_lib.run_local_search.restype = _CResult

_lib.run_simulated_annealing.argtypes = [
    ctypes.c_char_p,              # Graph filename
    ctypes.c_uint64,              # Seed
    ctypes.POINTER(ctypes.c_int8),# Initial partition pointer (or NULL)
    ctypes.c_double,              # Initial temperature
    ctypes.c_double,              # Cooling rate
    ctypes.c_int                  # Max iterations
]
_lib.run_simulated_annealing.restype = _CResult

# Approximations
_lib.run_randomized_half.argtypes = [
    ctypes.c_char_p,              # Graph filename
    ctypes.c_int,                 # Trials
    ctypes.c_uint64               # Seed
]
_lib.run_randomized_half.restype = _CResult

_lib.run_randomized_greedy_edges.argtypes = [
    ctypes.c_char_p,              # Graph filename
    ctypes.c_int,                 # Trials
    ctypes.c_uint64               # Seed
]
_lib.run_randomized_greedy_edges.restype = _CResult

_lib.run_randomized_greedy_vertices.argtypes = [
    ctypes.c_char_p,              # Graph filename
    ctypes.c_int,                 # Trials
    ctypes.c_uint64               # Seed
]
_lib.run_randomized_greedy_vertices.restype = _CResult


# 4. Helpers
def _get_c_filename(graph: Graph) -> bytes:
    """Helper to extract and encode graph filename."""
    path = graph.filename if hasattr(graph, 'filename') else getattr(graph, 'file_path', '')
    return path.encode('utf-8')


def _convert_partition_to_c(initial_result: Optional[Result], n: int):
    """
    Converts a Python Result 1-indexed dictionary {1: val, 2: val, ...}
    into a 0-indexed contiguous ctypes int8 C-array.
    """
    if initial_result is None or not initial_result.partition:
        return None
    
    init_list = [int(initial_result.partition[i]) for i in range(1, n + 1)]
    return (ctypes.c_int8 * n)(*init_list)


def _unpack_c_result(c_res: _CResult, n: int, tag: str, elapsed_time: float) -> Result:
    """Helper to convert C struct to Python Result object and free C memory."""
    final_partition = {i + 1: c_res.partition[i] for i in range(n)}
    weight = c_res.cut_weight
    edges = c_res.cut_edge_count

    _lib.free_c_result(c_res)

    return Result(
        partition=final_partition,
        cut_weight=weight,
        cut_edge_count=edges,
        type=tag,
        execution_time=elapsed_time
    )


# ============ HEURISTICS ==============

def local_search(
    graph: Graph, 
    seed: int = 42, 
    initial_result: Optional[Result] = None
) -> Result:
    start_time = time.perf_counter()
    n = graph.V_count
    c_filename = _get_c_filename(graph)
    c_init_partition = _convert_partition_to_c(initial_result, n)

    c_res = _lib.run_local_search(c_filename, seed, c_init_partition)
    elapsed_time = time.perf_counter() - start_time
    tag = "Local Search (Warm-Started)" if initial_result else "Local Search (C++)"

    return _unpack_c_result(c_res, n, tag, elapsed_time)


def simulated_annealing(
    graph: Graph,
    seed: int = 42,
    initial_result: Optional[Result] = None,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 100000
) -> Result:
    start_time = time.perf_counter()
    n = graph.V_count
    c_filename = _get_c_filename(graph)
    c_init_partition = _convert_partition_to_c(initial_result, n)

    c_res = _lib.run_simulated_annealing(
        c_filename, seed, c_init_partition,
        initial_temp, cooling_rate, max_iterations
    )
    elapsed_time = time.perf_counter() - start_time
    tag = "Simulated Annealing (Warm-Started)" if initial_result else "Simulated Annealing (C++)"

    return _unpack_c_result(c_res, n, tag, elapsed_time)


# ============ APPROXIMATIONS ==============

def randomized_half(graph: Graph, TRIALS: int = 200, seed: int = 42) -> Result:
    start_time = time.perf_counter()
    n = graph.V_count
    c_filename = _get_c_filename(graph)

    c_res = _lib.run_randomized_half(c_filename, TRIALS, seed)
    elapsed_time = time.perf_counter() - start_time
    tag = f"Randomized 0.5 (Best out of {TRIALS})"

    return _unpack_c_result(c_res, n, tag, elapsed_time)


def randomized_greedy_edges(graph: Graph, TRIALS: int = 100, seed: int = 42) -> Result:
    start_time = time.perf_counter()
    n = graph.V_count
    c_filename = _get_c_filename(graph)

    c_res = _lib.run_randomized_greedy_edges(c_filename, TRIALS, seed)
    elapsed_time = time.perf_counter() - start_time
    tag = f"Randomized Greedy Edges (Best out of {TRIALS})"

    return _unpack_c_result(c_res, n, tag, elapsed_time)


def randomized_greedy_vertices(graph: Graph, TRIALS: int = 100, seed: int = 42) -> Result:
    start_time = time.perf_counter()
    n = graph.V_count
    c_filename = _get_c_filename(graph)

    c_res = _lib.run_randomized_greedy_vertices(c_filename, TRIALS, seed)
    elapsed_time = time.perf_counter() - start_time
    tag = f"Randomized Greedy Vertices (Best out of {TRIALS})"

    return _unpack_c_result(c_res, n, tag, elapsed_time)