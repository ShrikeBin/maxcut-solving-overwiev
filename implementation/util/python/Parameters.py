from dataclasses import dataclass, field
from typing import Any

@dataclass
class Parameters:
    """Holds configuration parameters for a run method with pretty printing."""
    name: str = "Parameters"
    params: dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, name: str = "Parameters", **kwargs: Any):
        self.name = name
        self.params = kwargs

    def __str__(self) -> str:
        """Prints a clean visual readout of the execution parameters."""
        BOLD = "\033[1m"
        DIM = "\033[2m"
        CYAN = "\033[96m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        
        header_text = f" CONFIG [{self.name.upper()}] "
        width = max(len(header_text) + 4, 42)
        padded_header = f"{header_text:^{width-2}}"
        
        lines = [
            f"{BOLD}{BLUE}┌{'─' * (width - 2)}┐{RESET}",
            f"{BOLD}{BLUE}│{padded_header}│{RESET}",
            f"{BOLD}{BLUE}├{'─' * (width - 2)}┤{RESET}"
        ]
        
        if not self.params:
            lines.append(f"  {DIM}{CYAN}(No parameters specified){RESET}")
        else:
            for k, v in self.params.items():
                # Format value nicely if it's float or other types
                if isinstance(v, float):
                    val_str = f"{v:.6g}"
                else:
                    val_str = str(v)
                lines.append(f"  {BOLD}{CYAN}{k:<20}:{RESET} {MAGENTA}{val_str}{RESET}")
                
        lines.append(f"{BOLD}{BLUE}└{'─' * (width - 2)}┘{RESET}")
        
        return "\n".join(lines)