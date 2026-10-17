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
        BOLD = "\033[1m"
        DIM = "\033[2m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        MAGENTA = "\033[95m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        set_0 = [v for v, p in sorted(self.partition.items()) if p == 0 or p is False]
        set_1 = [v for v, p in sorted(self.partition.items()) if p == 1 or p is True]
        
        header_text = f" MAX-CUT RESULT [{self.type.upper()}] "
        width = max(len(header_text) + 4, 42)
        
        padded_header = f"{header_text:^{width-2}}"
        
        weight_str = f"{self.cut_weight:.4f}" if isinstance(self.cut_weight, float) and not self.cut_weight.is_integer() else f"{int(self.cut_weight)}"
        
        lines = [
            f"{BOLD}{MAGENTA}┌{'─' * (width - 2)}┐{RESET}",
            f"{BOLD}{MAGENTA}│{padded_header}│{RESET}",
            f"{BOLD}{MAGENTA}├{'─' * (width - 2)}┤{RESET}",
            f"  {BOLD}Total Cut Weight:{RESET}  {GREEN}{weight_str:<10}{RESET}",
            f"  {BOLD}Edges in Cut:{RESET}      {DIM}{GREEN}{self.cut_edge_count:<10}{RESET}",
            f"  {BOLD}Execution Time:{RESET}    {YELLOW}{self.execution_time:.6f}s{RESET}",
            f"{BOLD}{MAGENTA}├{'─' * (width - 2)}┤{RESET}",
            f"  {BOLD}{CYAN}● Partition 0{RESET} ({len(set_0)} nodes):",
            f"    {CYAN}{set_0}{RESET}",
            f"",
            f"  {BOLD}{YELLOW}● Partition 1{RESET} ({len(set_1)} nodes):",
            f"    {YELLOW}{set_1}{RESET}",
            f"{BOLD}{MAGENTA}└{'─' * (width - 2)}┘{RESET}"
        ]
        
        return "\n".join(lines)