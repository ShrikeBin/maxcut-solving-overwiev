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