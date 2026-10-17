import sys
import glob
import os
from itertools import product

from implementation.util.python.Graph import Graph
from implementation.util.python.Parameters import Parameters
import implementation.methodologies.heuristics_memetics.python.cpplib_heuristics_memetics as cpplibhm

# ===================================================================================================
GRID = {
    "max_iterations": [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000],
    "cooling_rate": [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999],
    "initial_temp": [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
}
SEED = 42
TOP_AMOUNT = 10
METHOD = "Simulated Annealing"
CSV_PATH = "all_results.csv"
# ===================================================================================================


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_annealing.py <path_or_glob_pattern>")
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

    # Generate all configurations once
    keys = GRID.keys()
    values = GRID.values()
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    print(f"Total configurations per graph: {len(combinations)}\n")

    for file_path in graph_files:
        print("==================================================")
        print(f"TESTING SIMULATED ANNEALING ON: {file_path}")
        print("==================================================\n")

        try:
            graph = Graph(file_path)
            print(graph)

            RED = "\033[31m"
            GREEN = "\033[92m"
            CYAN = "\033[96m"
            MAGENTA = "\033[95m"
            YELLOW = "\033[93m"
            RST = "\033[0m"

            # Helper execution function
            def run_config(tag, config):
                param_box = Parameters(name=tag, **config)
                print(param_box)

                res = cpplibhm.simulated_annealing(graph, seed=SEED, **config)
                elapsed = res.execution_time

                p_str = res.clear_partition()
                print(p_str)
                print("~" * 50)

                return {
                    "tag": tag,
                    "config": config,
                    "weight": res.cut_weight,
                    "time": elapsed,
                    "edges": res.cut_edge_count,
                    "print_str": p_str
                }

            # ==========================================
            # GRID SEARCH
            # ==========================================
            print(f"\n{MAGENTA}*** GRID SEARCH: Simulated Annealing ***{RST}\n")
            results_tracker = []

            for idx, config in enumerate(combinations, 1):
                tag = f"Simulated Annealing [Config #{idx}/{len(combinations)}]"
                results_tracker.append(run_config(tag, config))

            # ==========================================
            # LEADERBOARD & SUMMARY
            # ==========================================
            print()
            print("==================================================")
            print(f"RESULTS SUMMARY: {file_path}")
            print("==================================================\n")

            def print_leaderboard(title, tracker):
                top_x = sorted(tracker, key=lambda x: (-x["weight"], x["time"]))[:TOP_AMOUNT]
                box_lines = [
                    f"{MAGENTA}╔════════════════════════════════════════════════════════════╗{RST}",
                    f"{MAGENTA}║{RST} {CYAN}{title:^58}{RST} {MAGENTA}║{RST}",
                    f"{MAGENTA}╠════════════════════════════════════════════════════════════╝{RST}"
                ]
                for i, item in enumerate(top_x, 1):
                    cfg_str = ", ".join([f"{k}={v}" for k, v in item['config'].items()])
                    box_lines.append(f" {MAGENTA}#{i}{RST} -> Weight: {GREEN}{item['weight']}{RST} | Time: {CYAN}{item['time']:.4f}s{RST}")
                    box_lines.append(f"      Config: {YELLOW}{cfg_str}{RST}")
                    box_lines.append("\n")
                box_lines.append(f"{MAGENTA}╚════════════════════════════════════════════════════════════╝{RST}")
                print("\n".join(box_lines))

            print_leaderboard("TOP SIMULATED ANNEALING CONFIGURATIONS", results_tracker)
            print()

            import csv
            write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["method", "graph", "config_idx", "config", "weight", "cut_edges", "time_s"])
                for i, r in enumerate(results_tracker, 1):
                    cfg = ";".join(f"{k}={v}" for k, v in r["config"].items())
                    w.writerow([METHOD, os.path.basename(file_path), i, cfg, r["weight"], r["edges"], f"{r['time']:.6f}"])
            print(f"[+] Appended {len(results_tracker)} rows to {CSV_PATH}")

        except Exception as e:
            print(f"Error occurred: {e}")
            raise e