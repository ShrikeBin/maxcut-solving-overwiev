import sys
import glob
import os
import subprocess

from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result
import implementation.methodologies.solvers.python.Solve as mip
import implementation.methodologies.approximations.python.Approximate as aprx
import implementation.methodologies.heuristics_memetics.python.cpplib_heuristics_memetics as cpplibhm
import implementation.methodologies.approximations.python.cpplib_approximations as cpplibax
import implementation.methodologies.quantum.python.Quantum as qtm

def ensure_cpp_library_built():
    """Builds cpp libs using make if it does not exist."""
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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(base_dir, "methodologies", "approximations", "cpp")
    lib_path = os.path.join(cpp_dir, "lib", "libapprox.so")

    if not os.path.exists(lib_path):
        print("[!] Shared C++ library missing. Compiling with Makefile...")
        try:
            subprocess.run(["make"], cwd=cpp_dir, check=True, capture_output=True, text=True)
            print("[+] Makefile compilation successful.\n")
        except subprocess.CalledProcessError as e:
            print(f"[-] Makefile compilation failed:\n{e.stderr}", file=sys.stderr)
            sys.exit(1)



# -- CONFIG --

DO_FANCY = True
TOP_AMOUNT = 7

RUN_BASIC = True

RUN_WARM = True
GOEMANS_VERBOSE = False

RUN_MIP = False
MIP_TIME_LIMIT = 1800
MIP_VERBOSE = True
MIP_CALLBACK = [0.30, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03, 0.01]



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
                best_result = Result(cut_weight=0, partition=[], cut_edge_count=0)

                def run_and_track(tag, func, *args, prep_time=0.0, **kwargs):
                    res = func(*args, **kwargs)
                    
                    # Fallback if result object already tracks its own execution time internally
                    if hasattr(res, 'execution_time') and res.execution_time > 0:
                        elapsed = res.execution_time + prep_time
                    else:
                        raise RuntimeError

                    global best_result
                    if res.cut_weight > best_result.cut_weight:
                        best_result = res
                    
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

                if RUN_BASIC:
                    # --- Simple Randoms (C++) ---
                    run_and_track("Randomized 0.5", cpplibax.randomized_half, graph, TRIALS=2000)
                    run_and_track("Randomized Greedy Edges", cpplibax.randomized_greedy_edges, graph, TRIALS=1000)
                    run_and_track("Randomized Greedy Vertices", cpplibax.randomized_greedy_vertices, graph, TRIALS=1000)

                    # --- Heuristics (C++) ---
                    run_and_track("Local Search [Naive]", cpplibhm.local_search, graph, seed=42, mode=cpplibhm.LocalSearchMode.NAIVE_ONE_FLIP)
                    run_and_track("Local Search [K-Flip]", cpplibhm.local_search, graph, seed=42, mode=cpplibhm.LocalSearchMode.K_FLIP, k=2)
                    run_and_track("Local Search [Kernighan-Lin]", cpplibhm.local_search, graph, seed=42, mode=cpplibhm.LocalSearchMode.KERNIGHAN_LIN)
                    run_and_track("Simulated Annealing", cpplibhm.simulated_annealing, graph, seed=42, max_iterations=100000, cooling_rate=0.999, initial_temp=100)

                    # --- Advanced Metaheuristics (C++) ---
                    run_and_track("Tabu Search [Short-Term]", cpplibhm.tabu_search, graph, seed=42, max_iterations=5000, tabu_tenure=15, mode=cpplibhm.TabuMode.SHORT_TERM_ONLY)
                    run_and_track("Tabu Search [Short & Long]", cpplibhm.tabu_search, graph, seed=42, max_iterations=5000, tabu_tenure=15, mode=cpplibhm.TabuMode.SHORT_AND_LONG_TERM)
                    run_and_track("Variable Neighborhood Search", cpplibhm.variable_neighborhood_search, graph, seed=42, k_max=4, max_iterations=500)
                    run_and_track("GRASP Search", cpplibhm.grasp_search, graph, seed=42, max_iterations=30, alpha=0.3)

                    # --- Evolutionary & Memetic Algorithms (C++) ---
                    run_and_track("Genetic Algorithm", cpplibhm.genetic_algorithm, graph, seed=42, population_size=30, generations=100, mutation_rate=0.05)
                    run_and_track("Genetic Algorithm + LS memetic", cpplibhm.genetic_ls_algorithm, graph, seed=42, population_size=30, generations=100, mutation_rate=0.05)
                    run_and_track("Island Genetic Algorithm + LS memetic", cpplibhm.island_ls_algorithm, graph, seed=42, population_size=30, generations=100, migration_interval=10, mutation_rate=0.05)

                if RUN_WARM:
                    # --- Approximation (SDP) ---
                    warmResult = aprx.goemans_williamson(graph, RANDOM_SLICE_TRIALS=200, SOLVE_VERBOSE=GOEMANS_VERBOSE)
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
                    run_and_track("Warm Local Search [Naive]", cpplibhm.local_search, graph, seed=42, mode=cpplibhm.LocalSearchMode.NAIVE_ONE_FLIP, initial_result=warmResult, prep_time=gw_time)
                    run_and_track("Warm Local Search [K-Flip]", cpplibhm.local_search, graph, seed=42, mode=cpplibhm.LocalSearchMode.K_FLIP, k=2, initial_result=warmResult, prep_time=gw_time)
                    run_and_track("Warm Local Search [Kernighan-Lin]", cpplibhm.local_search, graph, seed=42, mode=cpplibhm.LocalSearchMode.KERNIGHAN_LIN, initial_result=warmResult, prep_time=gw_time)
                    run_and_track("Warm Simulated Annealing", cpplibhm.simulated_annealing, graph, seed=42, max_iterations=100000, initial_result=warmResult, cooling_rate=0.999, initial_temp=100, prep_time=gw_time)

                    # --- Warm-Started Advanced Metaheuristics (C++) ---
                    run_and_track("Warm Tabu Search [Short-Term]", cpplibhm.tabu_search, graph, seed=42, initial_result=warmResult, max_iterations=5000, tabu_tenure=15, mode=cpplibhm.TabuMode.SHORT_TERM_ONLY, prep_time=gw_time)
                    run_and_track("Warm Tabu Search [Short & Long]", cpplibhm.tabu_search, graph, seed=42, initial_result=warmResult, max_iterations=5000, tabu_tenure=15, mode=cpplibhm.TabuMode.SHORT_AND_LONG_TERM, prep_time=gw_time)
                    run_and_track("Warm Variable Neighborhood Search", cpplibhm.variable_neighborhood_search, graph, seed=42, initial_result=warmResult, k_max=4, max_iterations=500, prep_time=gw_time)

                    # --- Warm-Started Evolutionary & Memetic Algorithms ---
                    run_and_track("Warm Genetic Algorithm", cpplibhm.genetic_algorithm, graph, seed=42, initial_result=warmResult, population_size=30, generations=100, mutation_rate=0.05, prep_time=gw_time)
                    run_and_track("Warm Genetic Algorithm + LS memetic", cpplibhm.genetic_ls_algorithm, graph, seed=42, initial_result=warmResult, population_size=30, generations=100, mutation_rate=0.05, prep_time=gw_time)
                    run_and_track("Warm Island Genetic Algorithm + LS memetic", cpplibhm.island_ls_algorithm, graph, seed=42, initial_result=warmResult, population_size=30, generations=100, migration_interval=10, mutation_rate=0.05, prep_time=gw_time)

                # --- Exact MIP (with gap checkpoints) ---
                if RUN_MIP:
                    def mip_checkpoint(res, gap):
                        pct = f"{gap * 100:.4f}%"
                        p_str = res.clear_partition()

                        global best_result
                        if res.cut_weight > best_result.cut_weight:
                            best_result = res

                        results_tracker.append({
                            "tag": f"Exact MIP [checkpoint gap<={pct} ({res.cut_weight:.4f} : OPT : {res.cut_weight*(gap+1):.4f})]",
                            "weight": res.cut_weight,
                            "time": res.execution_time,
                            "edges": res.cut_edge_count,
                            "print_str": p_str
                        })
                        print(p_str)

                    run_and_track(
                        "Exact MIP [final]",
                        mip.solve_with_partial_results,
                        graph,
                        SOLVE_VERBOSE=MIP_VERBOSE,
                        time_limit=MIP_TIME_LIMIT,
                        gap_checkpoints=MIP_CALLBACK,
                        on_checkpoint=mip_checkpoint,
                    )

                # === FANCY PRINTING ===
                print()
                print()
                top_x = sorted(results_tracker, key=lambda x: (-x["weight"], x["time"]))[:TOP_AMOUNT]

                RED = "\033[31m"
                GREEN = "\033[92m"
                CYAN = "\033[96m"
                MAGENTA = "\033[95m"
                YELLOW = "\033[93m"
                RST = "\033[0m"

                print("==================================================")
                print(f"FINISHED SOLVING: {file_path}")
                print("==================================================\n")

                box_lines = [
                    f"{MAGENTA}╔════════════════════════════════════════════════════════════╗{RST}",
                    f"{MAGENTA}║{RST}                   {CYAN}RANKED TOP CUT RESULTS{RST}                   {MAGENTA}║{RST}",
                    f"{MAGENTA}╠════════════════════════════════════════════════════════════╝{RST}"
                ]
                
                for idx, item in enumerate(top_x, 1):
                    box_lines.append(f" {MAGENTA}#{idx}{RST} -> Method: {YELLOW}{item['tag']}{RST}")
                    box_lines.append(f"      Weight: {GREEN}{item['weight']}{RST} | Edges: {RED}{item['edges']}{RST} | Time: {CYAN}{item['time']:.9f}s{RST}")
                    box_lines.append("\n")
                
                box_lines.append(f"{MAGENTA}╚════════════════════════════════════════════════════════════╝{RST}")

                print("\n".join(box_lines))

                #graph.draw_interactive(partition=best_result.partition)
                
            except Exception as e:
                print(f"Error occurred: {e}")
                raise e
            
        else:
            print('Cmon.... run fancy...')
            print("\n")