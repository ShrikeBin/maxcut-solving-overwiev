import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from implementation.util.python.Graph import Graph
from implementation.util.python.Result import Result


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _edge_list(graph: Graph):
    """0-indexed (u, v, w) edges from the 1-indexed upper-triangular matrix."""
    n = graph.V_count
    return [(u - 1, v - 1, graph.matrix[u][v])
            for u in range(1, n + 1)
            for v in range(u + 1, n + 1)
            if graph.matrix[u][v] != 0.0]


def _cost_diagonal(n, edges):
    """cut weight of every bitstring; index i has qubit q = bit q of i (little-endian). -> Qskit different convetion"""
    idx = np.arange(2 ** n, dtype=np.uint32)
    diag = np.zeros(2 ** n)
    for u, v, w in edges:
        diag += w * (((idx >> u) ^ (idx >> v)) & 1)
    return diag


def _apply_mixer(psi, n, beta):
    """Apply RX(2*beta) = exp(-i*beta*X) on every qubit, in place."""
    c = float(np.cos(beta))
    s = complex(-1j * np.sin(beta))
    for q in range(n):
        v = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)  # view, psi is contiguous
        a = v[:, 0, :].copy()
        b = v[:, 1, :]
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = c * b + s * a


def _qaoa_state(params, n, diag, p):
    """Exact QAOA state: layers of exp(-i*gamma*C) then RX(2*beta) on all qubits."""
    gammas, betas = params[:p], params[p:]
    psi = np.full(2 ** n, 2.0 ** (-n / 2), dtype=np.complex64)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * diag).astype(np.complex64)
        _apply_mixer(psi, n, float(b))
    return psi


def _expected_cut(params, n, diag, p):
    psi = _qaoa_state(params, n, diag, p)
    return float(np.dot(psi.real ** 2 + psi.imag ** 2, diag))


def _build_circuit(n, edges, p):
    """
    Same unitary as _qaoa_state. exp(-i*gamma*w*(1-ZZ)/2) equals RZZ(-gamma*w)
    up to a global phase. Returns an unmeasured parametrized circuit.
    """
    gammas = [Parameter(f"g{i}") for i in range(p)]
    betas = [Parameter(f"b{i}") for i in range(p)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for u, v, w in edges:
            qc.rzz(-gammas[layer] * w, u, v)
        for q in range(n):
            qc.rx(2 * betas[layer], q)
    return qc, gammas, betas


def _binding(gammas, betas, params):
    p = len(gammas)
    return {**{gammas[i]: float(params[i]) for i in range(p)},
            **{betas[i]: float(params[p + i]) for i in range(p)}}


def _best_from_counts(counts, n, edges):
    """Returns (best_bits, mean_cut over all shots). bits[q] is qubit q."""
    best_bits, best_cut = None, -1.0
    total, shots = 0.0, 0
    for bitstring, count in counts.items():
        bits = bitstring[::-1]  # Qiskit is little-endian
        cut = sum(w for u, v, w in edges if bits[u] != bits[v])
        total += cut * count
        shots += count
        if cut > best_cut:
            best_cut, best_bits = cut, bits
    return best_bits, total / shots


def _get_backend(service, n, backend_name):
    if backend_name:
        return service.backend(backend_name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n)


# ----------------------------------------------------------------------------
# Step 1: check the circuit matches the numpy simulation (use a small graph)
# ----------------------------------------------------------------------------
def verify_circuit(graph: Graph, P_LAYERS: int = 1, seed: int = 0) -> bool:
    from qiskit.quantum_info import Statevector
    n = graph.V_count
    edges = _edge_list(graph)
    diag = _cost_diagonal(n, edges)
    params = np.random.default_rng(seed).uniform(0.1, 1.5, 2 * P_LAYERS)

    qc, g, b = _build_circuit(n, edges, P_LAYERS)
    bound = qc.assign_parameters(_binding(g, b, params))
    from_circuit = float(Statevector(bound).probabilities() @ diag)
    from_numpy = _expected_cut(params, n, diag, P_LAYERS)

    ok = bool(np.isclose(from_circuit, from_numpy, rtol=1e-4, atol=1e-3))
    print(f"[verify] circuit={from_circuit:.5f} numpy={from_numpy:.5f} match={ok}")
    return ok


# ----------------------------------------------------------------------------
# Step 2: reference numbers (exact optimum + random-sampling baseline)
# ----------------------------------------------------------------------------
def reference_values(graph: Graph, shots: int, seed: int = 0) -> dict:
    n = graph.V_count
    diag = _cost_diagonal(n, _edge_list(graph))
    rng = np.random.default_rng(seed)
    sample = diag[rng.integers(0, 2 ** n, size=shots)]
    return {"optimum": float(diag.max()),
            "random_best_of_shots": float(sample.max()),
            "random_mean": float(sample.mean())}


# ----------------------------------------------------------------------------
# Step 3 (optional, costs no QPU time): transpile for a device and inspect size
# ----------------------------------------------------------------------------
def transpile_report(graph: Graph, P_LAYERS: int, BACKEND_NAME=None, seed: int = 0) -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    n = graph.V_count
    edges = _edge_list(graph)
    qc, g, b = _build_circuit(n, edges, P_LAYERS)
    qc.measure_all()
    dummy = np.full(2 * P_LAYERS, 0.5)  # gate counts don't depend on angle values
    bound = qc.assign_parameters(_binding(g, b, dummy))

    backend = _get_backend(QiskitRuntimeService(), n, BACKEND_NAME)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                      seed_transpiler=seed)
    isa = pm.run(bound)
    report = {"backend": backend.name, "depth": isa.depth(),
              "two_qubit_gates": isa.num_nonlocal_gates()}
    print(f"[transpile_report] {report}")
    return report


# ----------------------------------------------------------------------------
# Main QAOA solver
# ----------------------------------------------------------------------------
def qaoa(graph: Graph,
         P_LAYERS: int,
         SHOTS: int,
         SOLVE_VERBOSE: bool,
         MODE: str = "simulator",       # "simulator" or "hardware"
         BACKEND_NAME=None,             # hardware only; None = least busy device
         N_RESTARTS: int = 3,
         MAX_ITER: int = 100,
         SEED: int = 0,
         MAX_SIMULATABLE_QUBITS: int = 24) -> Result:
    """
    QAOA for Max-Cut.
      1. Angles (gamma, beta) are optimized classically with COBYLA against the
         EXACT expectation (numpy statevector), with random restarts.
      2. The optimized circuit is sampled either on AerSimulator ("simulator")
         or in ONE job on an IBM device via Qiskit Runtime ("hardware").
    The best bitstring over all shots is returned; the mean cut over all shots
    is included in the result type string, so compare it with the random baseline.
    """
    if MODE not in ("simulator", "hardware"):
        raise ValueError("MODE must be 'simulator' or 'hardware'")

    start_time = time.perf_counter()
    n = graph.V_count
    if n > MAX_SIMULATABLE_QUBITS:
        raise ValueError(f"n={n} is too large for exact angle optimization "
                         f"(limit {MAX_SIMULATABLE_QUBITS}).")

    edges = _edge_list(graph)
    diag = _cost_diagonal(n, edges)
    rng = np.random.default_rng(SEED)
    w_scale = max((abs(w) for _, _, w in edges), default=1.0)

    # --- classical angle optimization (exact expectation, no shot noise) ---
    best_opt = None
    for r in range(N_RESTARTS):
        x0 = np.concatenate([rng.uniform(0.0, np.pi, P_LAYERS) / w_scale,
                             rng.uniform(0.0, np.pi / 2, P_LAYERS)])
        opt = minimize(lambda x: -_expected_cut(x, n, diag, P_LAYERS), x0,
                       method="COBYLA",
                       options={"maxiter": MAX_ITER, "rhobeg": 0.3})
        if SOLVE_VERBOSE:
            print(f"[qaoa] restart {r}: expected cut = {-opt.fun:.4f}")
        if best_opt is None or opt.fun < best_opt.fun:
            best_opt = opt
    if SOLVE_VERBOSE:
        print(f"[qaoa] best expected cut = {-best_opt.fun:.4f}, "
              f"exact optimum = {diag.max():.4f}")

    # --- build the final measured circuit with optimized angles ---
    qc, g, b = _build_circuit(n, edges, P_LAYERS)
    qc.measure_all()
    bound = qc.assign_parameters(_binding(g, b, best_opt.x))

    # --- sample ---
    if MODE == "simulator":
        sim = AerSimulator()
        counts = sim.run(transpile(bound, sim), shots=SHOTS,
                         seed_simulator=SEED).result().get_counts()
        where = "AerSimulator"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        backend = _get_backend(QiskitRuntimeService(), n, BACKEND_NAME)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                          seed_transpiler=SEED)
        isa = pm.run(bound)
        if SOLVE_VERBOSE:
            print(f"[qaoa] {backend.name}: depth={isa.depth()}, "
                  f"two-qubit gates={isa.num_nonlocal_gates()}")
        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa], shots=SHOTS)
        print(f"[qaoa] submitted job {job.job_id()} (retrievable later by id)")
        counts = job.result()[0].data.meas.get_counts()
        where = f"real hardware: {backend.name}"

    best_bits, mean_cut = _best_from_counts(counts, n, edges)
    partition = {i + 1: int(best_bits[i]) for i in range(n)}
    weight, edge_count = graph.get_cut_info(partition)

    return Result(
        partition=partition,
        cut_weight=weight,
        cut_edge_count=edge_count,
        type=f"QAOA (p={P_LAYERS}, {SHOTS} shots, {where}, mean cut {mean_cut:.2f})",
        execution_time=time.perf_counter() - start_time
    )