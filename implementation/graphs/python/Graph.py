class Graph:
    def __init__(self, filename):
        """
         Graph internally held as undirected with summed weights as upper triangular matrix
        """
        with open(filename, "r") as file:
            lines = file.readlines()
        
        self.isDirected = int(lines[0].split("=")[1].strip())
        self.isWeighted = int(lines[1].split("=")[1].strip())
        self.V_count = int(lines[2].split("=")[1].strip())
        self.E_count = int(lines[3].split("=")[1].strip())
        
        self.matrix = [[0.0] * (self.V_count + 1) for _ in range(self.V_count + 1)]
            
        for line in lines[4:]:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("/*") or clean_line.startswith("*/"):
                continue
                
            parts = clean_line.split(";")
            if len(parts) >= 2:
                u = int(parts[0].strip())
                v = int(parts[1].strip())
                
                weight = float(parts[2].strip()) if self.isWeighted else 1.0
               
                row = min(u, v)
                col = max(u, v)
                
                self.matrix[row][col] += weight
                        
    def get_cut_info(self, partition: dict[int, int]) -> tuple[float, int]:
        """
        Calculates the true weight of a cut given a vertex -> partition mapping.
        """
        cut_weight = 0.0
        edge_count = 0

        for u in range(1, self.V_count + 1):
            for v in range(u + 1, self.V_count + 1):
                val = self.matrix[u][v]
                if val != 0.0 and partition[u] != partition[v]:
                    cut_weight += val
                    edge_count += 1
                        
        return cut_weight, edge_count
    
    def __str__(self):
        BOLD = "\033[1m"
        DIM = "\033[2m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        output = [
            f"{BOLD}{CYAN}┌── Graph Properties ────────────────────┐{RESET}",
            f"  {BOLD}Directed:{RESET}  0 {DIM}(Collapsed from original: {self.isDirected}){RESET}",
            f"  {BOLD}Weighted:{RESET}  {GREEN if self.isWeighted else DIM}{self.isWeighted}{RESET}",
            f"  {BOLD}Nodes |V|:{RESET} {YELLOW}{self.V_count:<6}{RESET}", 
            f"  {BOLD}Edges |E|:{RESET} {YELLOW}{self.E_count:<6}{RESET}",
            f"{BOLD}{CYAN}├── Adjacency Matrix ────────────────────┤{RESET}"
        ]
        
        if(self.V_count <= 20):
            headers = f"     {BOLD}{CYAN}" + "  ".join(f"{i:2}" for i in range(1, self.V_count + 1)) + f"{RESET}"
            output.append(headers)
            
            for i in range(1, self.V_count + 1):
                row_str = f"  {BOLD}{CYAN}{i:2}:{RESET} "
                row_values = []
                
                for j in range(1, self.V_count + 1):
                    row = min(i, j)
                    col = max(i, j)
                    val = self.matrix[row][col] if i != j else 0.0
                    
                    if val == 0.0:
                        row_values.append(f"{DIM} 0{RESET}")
                    else:
                        formatted_val = f"{int(val):2}" if val.is_integer() else f"{val:.1f}"
                        row_values.append(f"{GREEN}{formatted_val}{RESET}")
                        
                row_str += "  ".join(row_values)
                output.append(row_str)
        else:
            output.append(f"{DIM}{CYAN}   [Too big for display]{RESET}")
            
        output.append(f"{BOLD}{CYAN}└────────────────────────────────────────┘{RESET}")
        return "\n".join(output)
    
    def draw(self, partition: dict[int, int] = None, node_size=200):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(1, self.V_count + 1))

        for u in range(1, self.V_count + 1):
            for v in range(u + 1, self.V_count + 1):
                weight = self.matrix[u][v]
                if weight != 0.0:
                    G.add_edge(u, v, weight=weight)

        pos = nx.spring_layout(G, seed=42)

        if partition:
            unique_partitions = list(set(partition.values()))
            cmap = plt.get_cmap("Set2")

            node_colors = [
                cmap(unique_partitions.index(partition[n]) / max(1, len(unique_partitions) - 1))
                if len(unique_partitions) > 1 else cmap(0)
                for n in G.nodes()
            ]

            cut_edges = []
            internal_edges = []
            for u, v in G.edges():
                if partition[u] != partition[v]:
                    cut_edges.append((u, v))
                else:
                    internal_edges.append((u, v))

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
            
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=internal_edges, 
                edge_color="black", 
                width=0.75, 
                alpha=0.2
            )
            
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=cut_edges, 
                edge_color="red", 
                style="dashed", 
                width=0.95, 
                alpha=0.7
            )
            
            nx.draw_networkx_labels(G, pos, font_weight="bold", font_size=8)
        else:
            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=node_size)
            nx.draw_networkx_edges(G, pos, edge_color="black", width=0.75, alpha=0.3)
            nx.draw_networkx_labels(G, pos, font_weight="bold", font_size=8)

        plt.tight_layout()
        plt.show()