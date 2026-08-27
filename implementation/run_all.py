import sys
import glob
import os
import subprocess

# REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)

# Graph class & Base Solvers
from implementation.graphs.python.Graph import Graph
import implementation.methodologies.solvers.python.Solve as mip
import implementation.methodologies.approximations.python.Approximate as aprx
import implementation.methodologies.heuristics_memetics.python.HeuristicsMemetics as hm



def ensure_cpp_library_built():
    """Builds libheuristics_memetics.so using make if it does not exist."""
    # Derive absolute path relative to where run_all.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(base_dir, "methodologies", "heuristics_memetics", "cpp")
    lib_path = os.path.join(cpp_dir, "lib", "libheuristics_memetics.so")

    if not os.path.exists(lib_path):
        print("[!] Shared C++ library missing. Compiling with Makefile...")
        try:
            subprocess.run(["make"], cwd=cpp_dir, check=True, capture_output=True, text=True)
            print("[+] Makefile compilation successful.\n")
        except subprocess.CalledProcessError as e:
            print(f"[-] Makefile compilation failed:\n{e.stderr}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <path_or_glob_pattern>")
        sys.exit(1)

    # 1. Compile C++ library if missing
    ensure_cpp_library_built()

    # 2. Parse CLI Arguments (Directories, Glob Patterns, or Direct Files)
    args = sys.argv[1:]
    graph_files = []
    
    for arg in args:
        if os.path.isdir(arg):
            search_path = os.path.join(arg, "*.graph")
            found_files = glob.glob(search_path)
            if not found_files:
                print(f"No `.graph` files found in '{arg}'")
            graph_files.extend(found_files)
        else:
            expanded = glob.glob(arg)
            if expanded:
                graph_files.extend([f for f in expanded if os.path.isfile(f)])
            elif os.path.isfile(arg):
                graph_files.append(arg)

    graph_files = sorted(list(set(graph_files)))

    if not graph_files:
        print(f"No valid `.graph` files found: {args}")
        sys.exit(1)
        
    print(f"Found {len(graph_files)} graph file(s)\n")
    
    # 3. Execution Pipeline
    for file_path in graph_files:
        print("==================================================")
        print(f"LOADING & SOLVING: {file_path}")
        print("==================================================\n")
        
        try:
            graph = Graph(file_path)
            print(graph)

            # --- Simple Randoms (C++) ---
            print(hm.randomized_half(graph, TRIALS=2000).clear_partition())
            print(hm.randomized_greedy_edges(graph, TRIALS=1000).clear_partition())
            print(hm.randomized_greedy_vertices(graph, TRIALS=1000).clear_partition())

            # --- Heuristics (C++) ---
            print(hm.local_search(graph, seed=42).clear_partition())
            print(hm.simulated_annealing(graph, seed=42, max_iterations=100000).clear_partition())

            # --- These take some time ---
            print(aprx.goemans_williamson(graph, RANDOM_SLICE_TRIALS=200, SOLVE_VERBOSE=True).clear_partition())
            print(mip.solve(graph, SOLVE_VERBOSE=True).clear_partition())
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        print("\n")