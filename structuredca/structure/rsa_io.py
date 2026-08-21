
# Imports ----------------------------------------------------------------------
import os.path
from typing import Dict

# RSA IO functions -------------------------------------------------------------
def write_rsa_map(file_path: str, rsa_map: Dict[str, float]) -> None:
    """Write an rsa_map {resid: str => RSA: float} to a cache file."""

    # Create output directory if it does not exist
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Stringify
    rsa_map_str = "\n".join(f"{resid} {rsa}" for resid, rsa in rsa_map.items()) + "\n"

    # Write
    with open(file_path, "w") as fs:
        fs.write(rsa_map_str)

def read_rsa_map(file_path: str) -> Dict[str, float]:
    """Read rsa_map cache file and return RSA mapping: {resid: str => RSA: float}."""

    # Guardians
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"ERROR in read_rsa_map(): RSA cache file file_path='{file_path}' does not exist.")

    # Parse and return
    COMMENT_CHAR = "#"
    rsa_map: Dict[str, float] = {}
    with open(file_path, "r") as fs:
        lines = [line.split() for line in fs.readlines() if len(line) >= 3 and line[0] != COMMENT_CHAR]
    for line in lines:
        if len(line) < 2: continue
        resid, rsa = line[0], line[1]
        rsa_map[resid] = float(rsa)

    # Guardian and return
    assert len(rsa_map) > 0, f"ERROR in read_rsa_map(): No RSA data found in file_path='{file_path}'."
    return rsa_map
