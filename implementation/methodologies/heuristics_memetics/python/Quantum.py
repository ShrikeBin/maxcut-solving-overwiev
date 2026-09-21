import time
import dimod
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from implementation.graphs.python.Graph import Graph
from implementation.graphs.python.Result import Result


def qaoa(graph: Graph,
        P_LAYERS: int,
        SHOTS: int,
        SOLVE_VERBOSE: bool,
        MAX_SIMULATABLE_QUBITS=20,
        REAL_BACKEND=None
        ) -> Result:
    """
    QAOA (Quantum Approximate Optimization Algorithm) approach to Max-Cut.
    Builds a p-layer QAOA circuit from the graph's edge structure, classically
    optimizes the (gamma, beta) angles to maximize expected cut size, then
    samples the optimized circuit and keeps the best observed bitstring.

    Cost unitary per edge (u,v): CX(u,v) -> RZ(2*gamma) on v -> CX(u,v)
    Mixer unitary per qubit:     RX(2*beta)

    Parameter optimization always runs on AerSimulator, regardless of
    REAL_BACKEND: COBYLA needs ~100 circuit evaluations, and submitting each
    as a separate hardware job would mean 100+ queued jobs just to find good
    angles. If REAL_BACKEND is set, exactly one final job -- using the angles
    already optimized on the simulator -- is submitted to that backend via
    qiskit_ibm_runtime. Requires device-specific service/account configuration
    and a qubit count within the device's physical limits.
    """
    start_time = time.perf_counter()
    n = graph.V_count

    if n > MAX_SIMULATABLE_QUBITS and REAL_BACKEND is None:
        print(f"[qaoa] WARNING: n={n} exceeds MAX_SIMULATABLE_QUBITS="
              f"{MAX_SIMULATABLE_QUBITS}. Statevector simulation will be "
              f"extremely slow/infeasible. Consider a subgraph, the analytical "
              f"p=1 formula, or real hardware instead.")

    # Build edge list from the graph's upper-triangular weight matrix.
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            weight = graph.matrix[u][v]
            if weight != 0.0:
                edges.append((u - 1, v - 1, weight))  # 0-indexed for qubits

    gammas = [Parameter(f"g{layer}") for layer in range(P_LAYERS)]
    betas = [Parameter(f"b{layer}") for layer in range(P_LAYERS)]

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for layer in range(P_LAYERS):
        for u, v, w in edges:
            qc.cx(u, v)
            qc.rz(2 * gammas[layer] * w, v)
            qc.cx(u, v)
        for q in range(n):
            qc.rx(2 * betas[layer], q)
    qc.measure(range(n), range(n))

    sim = AerSimulator()

    def expected_cut(params):
        binding = {**{gammas[i]: params[i] for i in range(P_LAYERS)},
                   **{betas[i]: params[P_LAYERS + i] for i in range(P_LAYERS)}}
        bound = qc.assign_parameters(binding)
        result = sim.run(bound, shots=SHOTS).result()
        counts = result.get_counts()
        total = 0.0
        for bitstring, count in counts.items():
            bits = bitstring[::-1]
            cut = sum(w for u, v, w in edges if bits[u] != bits[v])
            total += cut * count
        return -(total / SHOTS)  # negated: scipy minimizes

    x0 = np.random.uniform(0.1, np.pi - 0.1, 2 * P_LAYERS)
    opt = minimize(expected_cut, x0, method="COBYLA",
                    options={"maxiter": 100, "disp": SOLVE_VERBOSE})

    binding = {**{gammas[i]: opt.x[i] for i in range(P_LAYERS)},
               **{betas[i]: opt.x[P_LAYERS + i] for i in range(P_LAYERS)}}
    bound = qc.assign_parameters(binding)

    if REAL_BACKEND is not None:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        service = QiskitRuntimeService()
        backend = service.backend(REAL_BACKEND)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(bound)

        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = SHOTS
        job = sampler.run([isa_circuit])
        pub_result = job.result()[0]
        counts = pub_result.data.meas.get_counts()
    else:
        final_result = sim.run(bound, shots=SHOTS).result()
        counts = final_result.get_counts()

    best_partition = {}
    best_edge_count = -1
    best_weight = -1.0
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        partition = {i + 1: int(bits[i]) for i in range(n)}
        weight, edge_count = graph.get_cut_info(partition)
        if weight > best_weight:
            best_weight = weight
            best_edge_count = edge_count
            best_partition = partition

    elapsed_time = time.perf_counter() - start_time

    return Result(
        partition=best_partition,
        cut_weight=best_weight,
        cut_edge_count=best_edge_count,
        type=f"QAOA (p={P_LAYERS}, {SHOTS} shots, "
             f"{'real hardware: ' + REAL_BACKEND if REAL_BACKEND else 'AerSimulator'})",
        execution_time=elapsed_time
    )


def quantum_annealing(
    graph: Graph,
    NUM_READS: int,
    NUM_SWEEPS: int,
    SOLVE_VERBOSE: bool,
    USE_REAL_QPU=False
    ) -> Result:
    """
    Quantum annealing approach to Max-Cut via QUBO formulation.
    Builds Q = A - D (adjacency minus degree matrix), the standard QUBO/Ising
    encoding for Max-Cut, and hands it to a dimod BinaryQuadraticModel sampler.

    minimize x^T Q x  <=>  maximize cut size, for x in {0,1}^n

    Defaults to dimod's SimulatedAnnealingSampler (classical annealing proxy,
    always available). Set USE_REAL_QPU=True to submit to real D-Wave hardware
    via dwave-system's EmbeddingComposite(DWaveSampler()) instead -- requires
    a Leap account with API token access (Quantum Voyager Program for academic use).
    """
    start_time = time.perf_counter()
    n = graph.V_count

    # Build the BQM directly from the graph's upper-triangular weight matrix.
    bqm = dimod.BinaryQuadraticModel('BINARY')
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            weight = graph.matrix[u][v]
            if weight != 0.0:
                bqm.add_linear(u, -weight)
                bqm.add_linear(v, -weight)
                bqm.add_interaction(u, v, 2 * weight)

    if USE_REAL_QPU:
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample(bqm, num_reads=NUM_READS)
    else:
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS
        )

    if SOLVE_VERBOSE:
        print(sampleset)

    best = sampleset.first
    best_partition = {node: int(val) for node, val in best.sample.items()}

    weight, edge_count = graph.get_cut_info(best_partition)

    elapsed_time = time.perf_counter() - start_time

    return Result(
        partition=best_partition,
        cut_weight=weight,
        cut_edge_count=edge_count,
        type=f"Quantum Annealing ({'D-Wave QPU' if USE_REAL_QPU else 'Simulated'}, "
             f"{NUM_READS} reads)",
        execution_time=elapsed_time
    )