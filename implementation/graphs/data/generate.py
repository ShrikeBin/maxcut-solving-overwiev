import numpy as np

def generate_random_graph(
        num_vertices, 
        num_edges, 
        min_weight, 
        max_weight, 
        directed=True, 
        weighted=True):
    
    num_vertices = int(num_vertices)
    num_edges = int(num_edges)
    
    max_possible_edges = num_vertices * (num_vertices - 1) if directed else (num_vertices * (num_vertices - 1)) // 2
    if num_edges > max_possible_edges:
        raise ValueError(f"EDGES > MAX_POSSIBLE_EDGES ({max_possible_edges})")

    edge_structure = set()
    final_edges = []
    rng = np.random.default_rng()

    while len(edge_structure) < num_edges:
        needed = int(num_edges - len(edge_structure))
        
        u_nodes = rng.integers(1, num_vertices + 1, size=needed)
        v_nodes = rng.integers(1, num_vertices + 1, size=needed)
        
        for u, v in zip(u_nodes, v_nodes):
            if u == v:
                continue
                
            structural_key = (u, v) if directed else tuple(sorted((u, v)))
            
            if structural_key not in edge_structure:
                edge_structure.add(structural_key)
                if(weighted):
                    weight = round(float(rng.uniform(min_weight, max_weight)), 1)
                else:
                    weight = 1.0
                final_edges.append((u, v, weight))
                
            if len(edge_structure) == num_edges:
                break

    output = []
    output.append(f"# directed = {1 if directed else 0}")
    output.append(f"# weighted = {1 if weighted else 0}")
    output.append(f"# vertices = {num_vertices}")
    output.append(f"# edges    = {num_edges}")
    output.append("/*")
    
    for u, v, weight in sorted(final_edges):
        output.append(f"    {u};{v};{weight}")
            
    output.append("*/")
    
    return "\n".join(output)





# --- Configuration ---
VERTICES = 1000
EDGES = (VERTICES)*15
MIN_WEIGHT = 1.0
MAX_WEIGHT = 3.0
DIRECTED = False
WEIGHTED = False


g_string = generate_random_graph(VERTICES, EDGES, MIN_WEIGHT, MAX_WEIGHT, DIRECTED, WEIGHTED)
filename = f"exampleGraphs/N{VERTICES}E{int(EDGES)}.graph"
with open(filename, "w", encoding="utf-8") as file:
    file.write(g_string)

print(f"Graph generated and saved to: {filename}")