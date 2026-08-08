import sys
import glob
import os
from implementation.graphs.python.Graph import Graph
from implementation.methodologies.solvers.python.Solve import solve_mip
from implementation.methodologies.approximations.python.Approximate import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <path>")
        sys.exit(1)

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
    
    for file_path in graph_files:
        print("==================================================")
        print(f"LOADING & SOLVING: {file_path}")
        print("==================================================\n")
        
        try:
            graph = Graph(file_path)
            print(graph)
            #graph.draw()

            print(randomized_half(graph, TRIALS=200).clear_partition())
            print(randomized_greedy_edges(graph, TRIALS=100).clear_partition())
            print(randomized_greedy_vertices(graph, TRIALS=100).clear_partition())
            print(goemans_williamson(graph, RANDOM_SLICE_TRIALS=200, SOLVE_VERBOSE=True).clear_partition())
            print(solve_mip(graph, SOLVE_VERBOSE=True).clear_partition())
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        print("\n")