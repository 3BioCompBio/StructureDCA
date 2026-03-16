
# Imports ----------------------------------------------------------------------
from typing import Any

# Base functions ---------------------------------------------------------------
def is_convertable_to(input_object: Any, input_type: type) -> bool:
    """Return if input_object is convertable to input_type."""
    try:
        _ = input_type(input_object)
        return True
    except:
        return False
    
def format_str(input_value: Any, round_digit: int=3, max_length: int=30) -> str:
    """Format a python object to a uniformized string format."""
    if isinstance(input_value, float):
        return str(round(input_value, round_digit))
    elif isinstance(input_value, str):
        if len(input_value) > max_length:
            return f"'{input_value[:max_length]}...'"
        return f"'{input_value}'"
    else:
        return str(input_value)
    
def memory_str(n_bytes: int) -> str:
    """Return a human readable string for a memory size measure (input in bytes)."""
    if n_bytes / 1024**4 > 1.0:
        return f"{n_bytes / 1024**4:.3f} TB"
    if n_bytes / 1024**3 > 1.0:
        return f"{n_bytes / 1024**3:.3f} GB"
    elif n_bytes / 1024**2 > 1.0:
        return f"{n_bytes / 1024**2:.3f} MB"
    elif n_bytes / 1024 > 1.0:
        return f"{n_bytes / 1024:.3f} kB"
    else:
        return f"{n_bytes} B"
    
def time_str(n_sec: float) -> str:
    """Return a human readable string for a time measure (input in seconds)."""
    if n_sec / (60*60*24) > 1.0:
        return f"{n_sec / (60*60*24):.3f} d."
    elif n_sec / (60*60) > 1.0:
        return f"{n_sec / (60*60):.3f} h."
    elif n_sec / 60 > 1.0:
        return f"{n_sec / 60:.3f} min."
    else:
        return f"{n_sec:.3f} sec."
