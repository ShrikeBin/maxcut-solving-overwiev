class Graph:
    def __init__(self, filename, useAdjList=True):
        self.useAdjList = useAdjList
        
        with open(filename, "r") as file:
            lines = file.readlines()
        
        self.isDirected = int(lines[0].split("=")[1].strip())
        self.isWeighted = int(lines[1].split("=")[1].strip())
        self.V_count = int(lines[2].split("=")[1].strip())
        self.E_count = int(lines[3].split("=")[1].strip())
        
        # Use either adjacency list OR an adjacency matrix
        if self.useAdjList:
            self.adj_list = {i: [] for i in range(1, self.V_count + 1)}
        else:
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
                
                if self.useAdjList:
                    self.adj_list[u].append((v, weight))
                    if not self.isDirected:
                        self.adj_list[v].append((u, weight))
                else:
                    self.matrix[u][v] = weight
                    if not self.isDirected:
                        self.matrix[v][u] = weight
                        
    def get_cut_info(self, partition: dict[int, int]) -> tuple[float, int]:
        """
        Calculates the true weight of a cut given a vertex -> partition mapping.
        """
        cut_weight = 0.0
        edge_count = 0

        if self.useAdjList:
            for u, neighbors in self.adj_list.items():
                for v, weight in neighbors:
                    # Prevent double counting undirected edges
                    if not self.isDirected and u > v:
                        continue
                    if partition[u] != partition[v]:
                        cut_weight += weight
                        edge_count += 1
        else:
            for u in range(1, self.V_count + 1):
                start_v = 1 if self.isDirected else u + 1
                for v in range(start_v, self.V_count + 1):
                    val = self.matrix[u][v]
                    if val != 0.0 and partition[u] != partition[v]:
                        cut_weight += val
                        edge_count += 1
                        
        return cut_weight, edge_count
    
    def __str__(self):
        storage_type = "List" if self.useAdjList else "Matrix"
        
        output = [
            f"--- Graph Properties ---",
            f"Directed  = {self.isDirected}\n"
            f"Weighted  = {self.isWeighted}",
            f"     |V|  = {self.V_count}", 
            f"     |E|  = {self.E_count}",
            f"Stored as = {storage_type}",
            f"------------------------"
        ]
        
        if self.useAdjList:
            for vertex, neighbors in sorted(self.adj_list.items()):
                neighbor_strs = []
                for neighbor, weight in neighbors:
                    if self.isWeighted:
                        neighbor_strs.append(f"{neighbor}(w:{weight})")
                    else:
                        neighbor_strs.append(str(neighbor))
                output.append(f"[{vertex}] -> " + ", ".join(neighbor_strs))
        else:
            headers = "    " + "  ".join(f"{i:2}" for i in range(1, self.V_count + 1))
            output.append(headers)
            
            for i in range(1, self.V_count + 1):
                row_str = f"{i:2}: "
                row_values = []
                for j in range(1, self.V_count + 1):
                    val = self.matrix[i][j]
                    if val == 0.0:
                        row_values.append(" 0")
                    else:
                        row_values.append(f"{int(val):2}" if val.is_integer() else f"{val:.1f}")
                row_str += "  ".join(row_values)
                output.append(row_str)
            
        return "\n".join(output) + "\n------------------------"
    
    def draw(
        self,
        partition: dict[int, int] = None,
        node_size=200,
    ):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph() if self.isDirected else nx.Graph()
        G.add_nodes_from(range(1, self.V_count + 1))

        if self.useAdjList:
            for u, neighbors in self.adj_list.items():
                for v, weight in neighbors:
                    if not self.isDirected and u > v:
                        continue
                    G.add_edge(u, v, weight=weight)
        else:
            for u in range(1, self.V_count + 1):
                start_v = 1 if self.isDirected else u + 1
                for v in range(start_v, self.V_count + 1):
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
                alpha=0.2,
                arrows=False
            )
            
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=cut_edges, 
                edge_color="red", 
                style="dashed", 
                width=0.95, 
                alpha=0.7,
                arrows=False
            )
            
            nx.draw_networkx_labels(G, pos, font_weight="bold", font_size=8)
        else:
            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=node_size)
            
            nx.draw_networkx_edges(
                G, pos, 
                edge_color="black", 
                width=0.75, 
                alpha=0.3,
                arrows=False
            )
            
            nx.draw_networkx_labels(G, pos, font_weight="bold", font_size=8)

        # Removed the old text-based edge labels completely to save screen space
        plt.tight_layout()
        plt.show()