import sys
import glob
import os
import subprocess

from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result
import implementation.methodologies.solvers.python.Solve as mip
import implementation.methodologies.approximations.python.Approximate as aprx
import implementation.methodologies.heuristics_memetics.python.CPPLIB as cpplib
import implementation.methodologies.heuristics_memetics.python.Quantum as qtm

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


DO_FANCY = True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <path_or_glob_pattern>")
        sys.exit(1)

    ensure_cpp_library_built()

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

        if(DO_FANCY):
            try:
                graph = Graph(file_path)
                print(graph)

                results_tracker = []

                def run_and_track(tag, func, *args, prep_time=0.0, **kwargs):
                    res = func(*args, **kwargs)
                    
                    # Fallback if result object already tracks its own execution time internally
                    if hasattr(res, 'execution_time') and res.execution_time > 0:
                        elapsed = res.execution_time + prep_time
                    else:
                        raise RuntimeError

                    p_str = res.clear_partition()
                    results_tracker.append({
                        "tag": tag,
                        "weight": res.cut_weight,
                        "time": elapsed,
                        "edges": res.cut_edge_count,
                        "print_str": p_str
                    })
                    print(p_str)
                    return res

                # --- Simple Randoms (C++) ---
                run_and_track("Randomized 0.5", cpplib.randomized_half, graph, TRIALS=2000)
                run_and_track("Randomized Greedy Edges", cpplib.randomized_greedy_edges, graph, TRIALS=1000)
                run_and_track("Randomized Greedy Vertices", cpplib.randomized_greedy_vertices, graph, TRIALS=1000)

                # --- Heuristics (C++) ---
                run_and_track("Local Search [Naive]", cpplib.local_search, graph, seed=42, mode=cpplib.LocalSearchMode.NAIVE_ONE_FLIP)
                run_and_track("Local Search [K-Flip]", cpplib.local_search, graph, seed=42, mode=cpplib.LocalSearchMode.K_FLIP, k=2)
                run_and_track("Local Search [Kernighan-Lin]", cpplib.local_search, graph, seed=42, mode=cpplib.LocalSearchMode.KERNIGHAN_LIN)
                run_and_track("Simulated Annealing", cpplib.simulated_annealing, graph, seed=42, max_iterations=100000, cooling_rate=0.999, initial_temp=100)

                # --- Advanced Metaheuristics (C++) ---
                run_and_track("Tabu Search [Short-Term]", cpplib.tabu_search, graph, seed=42, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_TERM_ONLY)
                run_and_track("Tabu Search [Short & Long]", cpplib.tabu_search, graph, seed=42, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_AND_LONG_TERM)
                run_and_track("Variable Neighborhood Search", cpplib.variable_neighborhood_search, graph, seed=42, k_max=4, max_iterations=500)
                run_and_track("GRASP Search", cpplib.grasp_search, graph, seed=42, max_iterations=30, alpha=0.3)

                # --- Evolutionary & Memetic Algorithms (C++) ---
                run_and_track("Genetic Algorithm", cpplib.genetic_algorithm, graph, seed=42, population_size=30, generations=100, mutation_rate=0.05)
                run_and_track("Genetic Algorithm + KL memetic", cpplib.genetic_KL_algorithm, graph, seed=42, population_size=30, generations=100, mutation_rate=0.05)
                run_and_track("Island Genetic Algorithm + KL memetic", cpplib.island_kl_algorithm, graph, seed=42, population_size=30, generations=100, migration_interval=10, mutation_rate=0.05)

                # --- Approximation (SDP) ---
                warmResult = aprx.goemans_williamson(graph, RANDOM_SLICE_TRIALS=200, SOLVE_VERBOSE=False)
                gw_time = warmResult.execution_time

                gw_p_str = warmResult.clear_partition()
                results_tracker.append({
                    "tag": "Goemans-Williamson (SDP)",
                    "weight": warmResult.cut_weight,
                    "time": gw_time,
                    "edges": warmResult.cut_edge_count,
                    "print_str": gw_p_str
                })
                print(gw_p_str)

                # --- Warm-Started Heuristics (C++) [Includes GW prep time] --- 
                run_and_track("Warm Local Search [Naive]", cpplib.local_search, graph, seed=42, mode=cpplib.LocalSearchMode.NAIVE_ONE_FLIP, initial_result=warmResult, prep_time=gw_time)
                run_and_track("Warm Local Search [K-Flip]", cpplib.local_search, graph, seed=42, mode=cpplib.LocalSearchMode.K_FLIP, k=2, initial_result=warmResult, prep_time=gw_time)
                run_and_track("Warm Local Search [Kernighan-Lin]", cpplib.local_search, graph, seed=42, mode=cpplib.LocalSearchMode.KERNIGHAN_LIN, initial_result=warmResult, prep_time=gw_time)
                run_and_track("Warm Simulated Annealing", cpplib.simulated_annealing, graph, seed=42, max_iterations=100000, initial_result=warmResult, cooling_rate=0.999, initial_temp=100, prep_time=gw_time)

                # --- Warm-Started Advanced Metaheuristics (C++) ---
                run_and_track("Warm Tabu Search [Short-Term]", cpplib.tabu_search, graph, seed=42, initial_result=warmResult, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_TERM_ONLY, prep_time=gw_time)
                run_and_track("Warm Tabu Search [Short & Long]", cpplib.tabu_search, graph, seed=42, initial_result=warmResult, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_AND_LONG_TERM, prep_time=gw_time)
                run_and_track("Warm Variable Neighborhood Search", cpplib.variable_neighborhood_search, graph, seed=42, initial_result=warmResult, k_max=4, max_iterations=500, prep_time=gw_time)

                # --- Warm-Started Evolutionary & Memetic Algorithms ---
                run_and_track("Warm Genetic Algorithm", cpplib.genetic_algorithm, graph, seed=42, initial_result=warmResult, population_size=30, generations=100, mutation_rate=0.05, prep_time=gw_time)
                run_and_track("Warm Genetic Algorithm + KL memetic", cpplib.genetic_KL_algorithm, graph, seed=42, initial_result=warmResult, population_size=30, generations=100, mutation_rate=0.05, prep_time=gw_time)
                run_and_track("Warm Island Genetic Algorithm + KL memetic", cpplib.island_kl_algorithm, graph, seed=42, initial_result=warmResult, population_size=30, generations=100, migration_interval=10, mutation_rate=0.05, prep_time=gw_time)

                # --- Exact MIP ---
                # run_and_track("EXACT MUO", mip.solve, graph, SOLVE_VERBOSE = True)



                # === FANCY PRINTING ===
                print()
                print()
                top_5 = sorted(results_tracker, key=lambda x: (-x["weight"], x["time"]))[:5]

                RED = "\033[31m"
                GREEN = "\033[92m"
                CYAN = "\033[96m"
                MAGENTA = "\033[95m"
                YELLOW = "\033[93m"
                RST = "\033[0m"

                box_lines = [
                    f"{MAGENTA}╔════════════════════════════════════════════════════════════╗{RST}",
                    f"{MAGENTA}║{RST}                     {CYAN}TOP 5 CUT RESULTS{RST}                      {MAGENTA}║{RST}",
                    f"{MAGENTA}╠════════════════════════════════════════════════════════════╝{RST}"
                ]
                
                for idx, item in enumerate(top_5, 1):
                    box_lines.append(f" {MAGENTA}#{idx}{RST} -> Method: {YELLOW}{item['tag']}{RST}")
                    box_lines.append(f"      Weight: {GREEN}{item['weight']}{RST} | Edges: {RED}{item['edges']}{RST} | Time: {CYAN}{item['time']:.9f}s{RST}")
                    box_lines.append("\n")
                
                box_lines.append(f"{MAGENTA}╚════════════════════════════════════════════════════════════╝{RST}")

                print("\n".join(box_lines))
                
            except Exception as e:
                print(f"Error occurred: {e}")
                raise e
            
        else:
            try:
                graph = Graph(file_path)
                print(graph)

                # --- Simple Randoms (C++) ---
                print(cpplib.randomized_half(graph, TRIALS=2000).clear_partition())
                print(cpplib.randomized_greedy_edges(graph, TRIALS=1000).clear_partition())
                print(cpplib.randomized_greedy_vertices(graph, TRIALS=1000).clear_partition())

                # --- Heuristics (C++) ---
                print(cpplib.local_search(graph, seed=42, mode=cpplib.LocalSearchMode.NAIVE_ONE_FLIP).clear_partition())
                print(cpplib.local_search(graph, seed=42, mode=cpplib.LocalSearchMode.K_FLIP, k=2).clear_partition())
                print(cpplib.local_search(graph, seed=42, mode=cpplib.LocalSearchMode.KERNIGHAN_LIN).clear_partition())
                print(cpplib.simulated_annealing(graph, seed=42, max_iterations=100000, cooling_rate=0.999, initial_temp=100).clear_partition())

                # --- Advanced Metaheuristics (C++) ---
                print(cpplib.tabu_search(graph, seed=42, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_TERM_ONLY).clear_partition())
                print(cpplib.tabu_search(graph, seed=42, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_AND_LONG_TERM).clear_partition())
                print(cpplib.variable_neighborhood_search(graph, seed=42, k_max=4, max_iterations=500).clear_partition())
                print(cpplib.grasp_search(graph, seed=42, max_iterations=30, alpha=0.3).clear_partition())

                # --- Evolutionary & Memetic Algorithms (C++) ---
                print(cpplib.genetic_algorithm(graph, seed=42, population_size=30, generations=100, mutation_rate=0.05).clear_partition())
                print(cpplib.genetic_KL_algorithm(graph, seed=42, population_size=30, generations=100, mutation_rate=0.05).clear_partition())
                print(cpplib.island_kl_algorithm(graph, seed=42, population_size=30, generations=100, migration_interval=10, mutation_rate=0.05).clear_partition())

                # --- Approximation (SDP) ---
                warmResult = aprx.goemans_williamson(graph, RANDOM_SLICE_TRIALS=200, SOLVE_VERBOSE=False)
                print(warmResult.clear_partition())

                # --- Warm-Started Heuristics (C++) --- 
                print(cpplib.local_search(graph, seed=42, mode=cpplib.LocalSearchMode.NAIVE_ONE_FLIP, initial_result=warmResult).clear_partition())
                print(cpplib.local_search(graph, seed=42, mode=cpplib.LocalSearchMode.K_FLIP, k=2, initial_result=warmResult).clear_partition())
                print(cpplib.local_search(graph, seed=42, mode=cpplib.LocalSearchMode.KERNIGHAN_LIN, initial_result=warmResult).clear_partition())
                print(cpplib.simulated_annealing(graph, seed=42, max_iterations=100000, initial_result = warmResult, cooling_rate=0.999, initial_temp=100).clear_partition())

                # --- Warm-Started Advanced Metaheuristics (C++) ---
                print(cpplib.tabu_search(graph, seed=42, initial_result=warmResult, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_TERM_ONLY).clear_partition())
                print(cpplib.tabu_search(graph, seed=42, initial_result=warmResult, max_iterations=5000, tabu_tenure=15, mode=cpplib.TabuMode.SHORT_AND_LONG_TERM).clear_partition())
                print(cpplib.variable_neighborhood_search(graph, seed=42, initial_result=warmResult, k_max=4, max_iterations=500).clear_partition())

                # --- Warm-Started Evolutionary & Memetic Algorithms (using Goemans-Williamson) ---
                print(cpplib.genetic_algorithm(graph, seed=42, initial_result=warmResult, population_size=30, generations=100, mutation_rate=0.05).clear_partition())
                print(cpplib.genetic_KL_algorithm(graph, seed=42, initial_result=warmResult, population_size=30, generations=100, mutation_rate=0.05).clear_partition())
                print(cpplib.island_kl_algorithm(graph, seed=42, initial_result=warmResult, population_size=30, generations=100, migration_interval=10, mutation_rate=0.05).clear_partition())


                # --- These take some time ---
                # print(mip.solve(graph, SOLVE_VERBOSE=True).clear_partition())
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            print("\n")