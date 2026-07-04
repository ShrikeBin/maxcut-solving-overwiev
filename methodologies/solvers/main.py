import sys
import glob
from src.Graph import Graph
from src.Solve import solve_mip

if __name__ == "__main__":
    graph_files_pattern = "../../exampleGraphs/graphs/*.graph"
    
    graph_files = sorted(glob.glob(graph_files_pattern))
    
    if not graph_files:
        print(f"No .graph files matching: {graph_files_pattern}")
        sys.exit(1)
        
    print(f"Found {len(graph_files)} graph file(s)\n")
    
    for file_path in graph_files:
        print(f"==================================================")
        print(f"LOADING & SOLVING: {file_path}")
        print(f"==================================================")
        
        graph = Graph(file_path, useAdjList=True)

        solve_mip(graph, PRINT_RESULTS=True, PRINT_EDGES=False, SOLVE_VERBOSE= False)
        print("\n")