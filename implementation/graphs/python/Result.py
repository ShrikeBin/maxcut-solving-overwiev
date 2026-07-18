from dataclasses import dataclass, replace

@dataclass
class Result:
    """Holds the partition mapping, total cut weight, algorithm variant, and execution time."""
    partition: dict[int, int]
    cut_edge_count: int
    cut_weight: float = 0.0
    type: str = "Unspecified"
    execution_time: float = 0.0
    
    def clear_partition(self) -> 'Result':
        """Returns a fresh copy of this Result instance with an empty partition dict."""
        return replace(self, partition={})
    
    def __str__(self) -> str:
        """Prints a clean visual readout of the cut strategy results."""
        set_0 = [v for v, p in sorted(self.partition.items()) if p == 0]
        set_1 = [v for v, p in sorted(self.partition.items()) if p == 1]
        
        return (
            f"=== Max-Cut Result [{self.type}] ===\n"
            f"Total Cut Weight: {self.cut_weight}\n"
            f"Edges in Cut: {self.cut_edge_count}\n"
            f"Execution Time:   {self.execution_time:.6f} seconds\n"
            f"Partition 0 ({len(set_0)} nodes): {set_0}\n"
            f"Partition 1 ({len(set_1)} nodes): {set_1}\n"
            f"=========================================="
        )