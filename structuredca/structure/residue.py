
# Imports ----------------------------------------------------------------------
from typing import Union, Dict
import numpy as np
from numpy.typing import NDArray
from Bio.PDB.Chain import Chain as BPChain
from Bio.PDB.Residue import Residue as BPResidue
from structuredca.sequence import AminoAcid

# Main -------------------------------------------------------------------------
class Residue:
    """Container class for a PDB residue.
    
    usage:
    res = Residue('A', '113', AminoAcid('K'))
    """

    # Constants ----------------------------------------------------------------

    # Maps aa-types to knowledge-based maximum surface area
    # Taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/#pone-0080635-t001
    MAX_SURFACE_MAP = {
        "ALA": 1.29,
        "ARG": 2.74,
        "ASN": 1.95,
        "ASP": 1.93,
        "CYS": 1.67,
        "GLN": 2.23,
        "GLU": 2.25,
        "GLY": 1.04,
        "HIS": 2.24,
        "ILE": 1.97,
        "LEU": 2.01,
        "LYS": 2.36,
        "MET": 2.24,
        "PHE": 2.40,
        "PRO": 1.59,
        "SER": 1.55,
        "THR": 1.55,
        "TRP": 2.85,
        "TYR": 2.63,
        "VAL": 1.74,
    }
    MAX_SURFACE_DEFAULT = 2.01 # mean value

    # Atoms values
    BACKBONE_ATOMS = ["N", "H", "H1", "H2", "H3", "1H", "2H", "3H", "CA", "HA", "C", "O", "OXT"]
    GLY_BACKBONE_ATOMS = ["N", "H", "H1", "H2", "H3", "1H", "2H", "3H", "C", "O", "OXT"] # keep C-apha since GLY has no side-chains
    HYDROGEN_ATOMS_PREFIXES = ["H", "1H", "2H", "3H"]

    # Constroctor --------------------------------------------------------------
    def __init__(
            self, 
            chain: str,
            position: str,
            amino_acid: AminoAcid,
            coords: NDArray[np.float32],
            rsa: Union[None, float]=None,
            plddt: Union[None, float]=None,
        ):

        # Guardians
        assert len(chain) == 1 and chain != " ", f"ERROR in Residue(): invalid chain='{chain}'."
        if rsa is not None:
            assert rsa >= 0.0, f"ERROR in Residue(): rsa='{rsa}' should be positive."

        # Set properties
        self.chain = str(chain)
        self.position = str(position)
        self.amino_acid = amino_acid
        self.rsa = rsa
        self.plddt = plddt

        # Set coordinates
        self.coords: NDArray[np.float32]
        if isinstance(coords, np.ndarray):
            self.coords = coords
        else:
            self.coords = np.array(coords, dtype=np.float32)
        if self.coords.size == 0:
            self.coords = np.zeros((0, 3), dtype=np.float32)
        assert self.coords.ndim == 2, f"ERROR in Residue(): input coords='{self.coords}' ({self.coords.shape}) should be an array of dim=2."
        assert self.coords.shape[1] == 3, f"ERROR in Residue(): input coords='{self.coords}' ({self.coords.shape}) should be of shape (n, 3)."
        if not self.coords.dtype != np.float32:
            self.coords = self.coords.astype(np.float32)

    # Properties ---------------------------------------------------------------
    @property
    def resid(self) -> str:
        return self.chain + self.position

    def __str__(self) -> str:
        rsa_str = "None" if self.rsa is None else f"{self.rsa:.2f}"
        plddt_str = "None" if self.plddt is None else f"{self.plddt:.2f}"
        return f"Residue('{self.resid}', '{self.amino_acid.three}', RSA={rsa_str}, pLDDT={plddt_str})"

    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return self.coords.shape[0]

    @property
    def ncoords(self) -> int:
        """Number of atom coordinates in the residue (=len(self))."""
        return self.coords.shape[0]

    # BioPython link -----------------------------------------------------------
    @classmethod
    def from_biopython_residue(
            cls,
            bp_residue: BPResidue,
            ignore_backbone_atoms: bool=True,
            rsa_map: Union[None, Dict[str, Union[float, None]]]=None,
        ) -> "Residue":
        """Return a StructureDCA Residues object from a BioPython Residue object."""

        # Get properties
        bp_chain: BPChain = bp_residue.get_parent()
        chain_id = str(bp_chain.id)
        position = str(bp_residue.id[1]) + str(bp_residue.id[2]).replace(" ", "")
        resid = chain_id + position
        amino_acid = AminoAcid.parse_three(str(bp_residue.get_resname()))
        if rsa_map is not None:
            rsa = rsa_map.get(resid, None)
        else:
            rsa = cls._get_bp_residue_rsa(bp_residue)
        plddt = cls._get_bp_residue_plddt(bp_residue)

        # Get coords
        bp_atoms = bp_residue.child_list
        if ignore_backbone_atoms:
            current_backbone_atoms = cls.GLY_BACKBONE_ATOMS if amino_acid.three_standard == "GLY" else cls.BACKBONE_ATOMS
            bp_atoms = [bp_atom for bp_atom in bp_atoms if bp_atom.id not in current_backbone_atoms]
        coords = [bp_atom.coord for bp_atom in bp_atoms]

        # Construct and return Residue
        return Residue(
            chain_id,
            position,
            amino_acid,
            coords=coords,
            rsa=rsa,
            plddt=plddt,
        )

    # Dependencies -------------------------------------------------------------
    @classmethod
    def _get_bp_residue_rsa(cls, bp_residue: BPResidue) -> Union[float, None]:
        """Get RSA of a BioPython Residue (or None if bp_residue.sasa is not assigned).
        - using biopython assigned SASA and by-AA Max surface table
        """
        sasa = bp_residue.sasa
        if sasa is None:
            return None
        aa_three = bp_residue.resname
        aa_three_standardized = AminoAcid._NON_STANDARD_AAS.get(aa_three, aa_three)
        max_surf = cls.MAX_SURFACE_MAP.get(aa_three_standardized, cls.MAX_SURFACE_DEFAULT)
        return float(sasa) / max_surf
    
    @classmethod
    def _get_bp_residue_plddt(cls, bp_residue: BPResidue) -> float:
        """Get pLDDT (or B-factor) of a BioPython Residue.
        - for B-factor, consider average on all atoms
        - for pLDDT, all atom have the same value
        """
        plddt_arr = [atom.bfactor for atom in bp_residue.get_atoms()]
        return round(float(np.mean(plddt_arr)), 2)
