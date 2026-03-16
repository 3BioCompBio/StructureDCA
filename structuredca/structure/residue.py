
# Imports ----------------------------------------------------------------------
from typing import List, Union
from structuredca.sequence import AminoAcid

# Main -------------------------------------------------------------------------
class Residue:
    """Container class for a PDB residue.
    
    usage:
    res = Residue('A', '113', AminoAcid('K'))
    """

    def __init__(
            self, 
            chain: str,
            position: str,
            amino_acid: AminoAcid,
            coords: Union[None, List[List[float]]]=None,
            rsa: Union[None, float]=None,
            plddt: Union[None, float]=None,
        ):

        # Guardians
        assert len(chain) == 1 and chain != " ", f"ERROR in Residue(): invalid chain='{chain}'."
        if rsa is not None:
            assert rsa >= 0.0, f"ERROR in Residue(): rsa='{rsa}' should be positive."

        # Set properties
        self.chain = chain
        self.position = position
        self.amino_acid = amino_acid
        self.coords: List[List[float]] = coords if coords is not None else []
        self.rsa = rsa
        self.plddt = plddt

    @property
    def resid(self) -> str:
        return self.chain + self.position

    def __str__(self) -> str:
        rsa_str = "None" if self.rsa is None else f"{self.rsa:.2f}"
        plddt_str = "None" if self.plddt is None else f"{self.plddt:.2f}"
        return f"Residue('{self.resid}', '{self.amino_acid.three}', RSA={rsa_str}, pLDDT={plddt_str})"
    
    def __len__(self) -> int:
        return len(self.coords)