import sys
import re
import csv
import glob
import os
import argparse

from implementation.graphs.python.Graph import Graph
import implementation.methodologies.heuristics_memetics.python.Quantum as qtm

MAX_QUBITS = 20  # graphs larger than this are skipped
MEAN_RE = re.compile(r"mean cut ([0-9.]+)")

CSV_FIELDS = ["graph", "n", "edges", "p", "kind", "backend", "shots", "mean_cut",
              "best_cut", "optimum", "random_mean", "two_qubit_gates", "depth", "time_s"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run QAOA on small Max-Cut graphs: noiseless simulator and, optionally, real IBM hardware."
    )
    ap.add_argument("paths", nargs="+", help="directories, files or glob patterns of .graph files")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2],
                    help="QAOA depths p for the simulator (default: 1 2)")
    ap.add_argument("--hw-layers", type=int, nargs="+", default=[1],
                    help="QAOA depths p for real hardware (default: 1)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--restarts", type=int, default=3, help="random restarts for angle optimization")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="check circuit == numpy simulation (graphs with n <= 20)")
    ap.add_argument("--hardware", action="store_true",
                    help="also run on a real IBM device (asks for confirmation before each job)")
    ap.add_argument("--backend", default=None,
                    help="IBM backend name; default = least busy device with enough qubits")
    ap.add_argument("--yes", action="store_true", help="skip the hardware confirmation prompt")
    ap.add_argument("--max-2q-gates", type=int, default=1000,
                    help="skip hardware if the transpiled circuit has more two-qubit gates (default: 1000)")
    return ap.parse_args()


def collect_graph_files(paths):
    graph_files = []
    for arg in paths:
        if os.path.isdir(arg):
            found = glob.glob(os.path.join(arg, "*.graph"))
            if not found:
                print(f"No `.graph` files found in '{arg}'")
            graph_files.extend(found)
        else:
            expanded = glob.glob(arg)
            if expanded:
                graph_files.extend([f for f in expanded if os.path.isfile(f)])
            elif os.path.isfile(arg):
                graph_files.append(arg)
    return sorted(set(graph_files))

def fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)

def print_summary(all_rows):
    RED = "\033[31m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    RST = "\033[0m"

    # group sim/hw rows that share graph and p
    groups = {}
    for r in all_rows:
        groups.setdefault((r["graph"], r["p"]), {})[r["kind"]] = r

    print()
    print(f"{MAGENTA}{'=' * 96}{RST}")
    print(f"{CYAN}QAOA SUMMARY (mean cut over all shots; 'kept' = hardware gain over random / simulator gain over random){RST}")
    print(f"{MAGENTA}{'=' * 96}{RST}")
    head = (f"{'graph':<18}{'E':>4}{'p':>3}{'optimum':>9}{'random':>8}{'sim':>8}"
            f"{'hw':>8}{'kept':>8}{'2q gates':>10}{'sim best':>10}{'hw best':>9}")
    print(head)
    print("-" * len(head))
    for (graph, p), g in sorted(groups.items()):
        sim, hw = g.get("sim"), g.get("hw")
        ref = sim or hw
        kept = None
        if sim and hw and sim["mean_cut"] - sim["random_mean"] > 1e-9:
            kept = (hw["mean_cut"] - hw["random_mean"]) / (sim["mean_cut"] - sim["random_mean"])
        print(f"{graph:<18}{ref['edges']:>4}{p:>3}{ref['optimum']:>9.1f}{ref['random_mean']:>8.4f}"
              f"{fmt(sim['mean_cut'] if sim else None):>8}{fmt(hw['mean_cut'] if hw else None):>8}"
              f"{fmt(kept, '.0%') if kept is not None else '-':>8}"
              f"{(hw['two_qubit_gates'] if hw else '-')!s:>10}"
              f"{fmt(sim['best_cut'] if sim else None, '.0f'):>10}{fmt(hw['best_cut'] if hw else None, '.0f'):>9}")
    print()


if __name__ == "__main__":
    args = parse_args()

    graph_files = collect_graph_files(args.paths)
    if not graph_files:
        print(f"No valid `.graph` files found: {args.paths}")
        sys.exit(1)

    sim_layers = set(args.layers)
    if args.hardware:
        sim_layers |= set(args.hw_layers)  # every hardware row gets a matching simulator row
    sim_layers = sorted(sim_layers)

    print(f"Found {len(graph_files)} graph file(s)\n")
    all_rows = []

    for file_path in graph_files:
        print("==================================================")
        print(f"LOADING & SOLVING: {file_path}")
        print("==================================================\n")

        try:
            graph = Graph(file_path)
            print(graph)

            n = graph.V_count
            if n > MAX_QUBITS:
                print(f"[skip] n={n} > {MAX_QUBITS} qubits, not suitable for this test.\n")
                continue

            name = os.path.basename(file_path).replace(".graph", "")
            edges = qtm._edge_list(graph)
            random_mean = sum(w for _, _, w in edges) / 2.0  # exact mean cut of a uniform random bitstring

            # --- Reference numbers ---
            ref = qtm.reference_values(graph, args.shots, seed=args.seed)
            optimum = ref["optimum"]
            print(f"[reference] exact optimum         = {optimum}")
            print(f"[reference] random mean (exact)   = {random_mean:.4f}")
            print(f"[reference] random best-of-{args.shots:<5} = {ref['random_best_of_shots']}\n")

            # --- Optional circuit sanity check ---
            if args.verify:
                qtm.verify_circuit(graph, P_LAYERS=1, seed=args.seed)
                print()

            graph_rows = []

            def run_qaoa(p, mode, backend=None, hw_info=None):
                res = qtm.qaoa(
                    graph, P_LAYERS=p, SHOTS=args.shots, SOLVE_VERBOSE=True,
                    MODE=mode, BACKEND_NAME=backend,
                    N_RESTARTS=args.restarts, SEED=args.seed,
                )
                m = MEAN_RE.search(getattr(res, "type", ""))
                mean_cut = float(m.group(1)) if m else float("nan")
                kind = "sim" if mode == "simulator" else "hw"
                print(f"[QAOA {kind} p={p}] {getattr(res, 'type', '')}")
                print(res.clear_partition())
                row = {
                    "graph": name, "n": n, "edges": len(edges), "p": p, "kind": kind,
                    "backend": backend or "AerSimulator", "shots": args.shots,
                    "mean_cut": mean_cut, "best_cut": res.cut_weight,
                    "optimum": optimum, "random_mean": random_mean,
                    "two_qubit_gates": hw_info["two_qubit_gates"] if hw_info else "",
                    "depth": hw_info["depth"] if hw_info else "",
                    "time_s": round(res.execution_time, 3),
                }
                graph_rows.append(row)
                return row

            # --- QAOA on AerSimulator (noiseless) ---
            for p in sim_layers:
                run_qaoa(p, "simulator")

            # --- QAOA on real IBM hardware ---
            if args.hardware:
                for p in args.hw_layers:
                    try:
                        print(f"\n[hardware] transpiling p={p} circuit for gate-count check...")
                        report = qtm.transpile_report(graph, p, args.backend, seed=args.seed)
                        print(f"[hardware] {report['two_qubit_gates']} two-qubit gates, "
                              f"depth {report['depth']} on {report['backend']}")

                        if report["two_qubit_gates"] > args.max_2q_gates:
                            print(f"[hardware] skipped: {report['two_qubit_gates']} gates > "
                                  f"--max-2q-gates {args.max_2q_gates}.")
                            continue

                        go = args.yes
                        if not go:
                            go = input("Submit this as ONE job to the real device? [y/N] ").strip().lower() == "y"
                        if not go:
                            print("[hardware] skipped.")
                            continue

                        run_qaoa(p, "hardware", backend=report["backend"], hw_info=report)
                    except Exception as e:
                        print(f"[hardware] p={p} failed: {e}")

            all_rows.extend(graph_rows)

        except Exception as e:
            print(f"Error occurred: {e}")
            raise e

    if all_rows:
        print_summary(all_rows)