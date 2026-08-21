
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Dict, Union
import warnings
import numpy as np
from numpy.typing import NDArray
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Structure import Structure as BPStructure
from Bio.PDB.Model import Model as BPModel
from Bio.PDB.Residue import Residue as BPResidue
from Bio.PDB.SASA import ShrakeRupley
from structuredca.utils import Logger
from structuredca.sequence import AminoAcid, Sequence, PairwiseAlignment
from structuredca.structure import Residue, StructureReader, read_rsa_map, write_rsa_map


# Main -------------------------------------------------------------------------

class Structure:
    """Structure object for parsing all Residues from ATOM lines and assign RSA (with biopython, DSSP or MuSiC) and distance matrix."""


    # Constants ----------------------------------------------------------------

    # Input validation constants
    ACCEPTED_CHAIN_DESCRIPTORS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    ACCEPTED_EXTENTIONS = StructureReader.ACCEPTED_EXTENTIONS

    # Atoms values
    BACKBONE_ATOMS = Residue.BACKBONE_ATOMS
    GLY_BACKBONE_ATOMS = Residue.GLY_BACKBONE_ATOMS
    HYDROGEN_ATOMS_PREFIXES = Residue.HYDROGEN_ATOMS_PREFIXES


    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            pdb_path: Union[str, Sequence],
            target_chains: str,
            homomeric_chains: Union[None, str]=None,
            ignore_hydrogen_atoms: bool=True,
            ignore_backbone_atoms: bool=True,
            distance_cache_path: Union[None, str]=None,
            solve_rsa: bool=False,
            solve_distances: bool=True,
            rsa_cache_path: Union[None, str]=None,
            logger: Union[bool, Logger]=False,
        ):
        """Structure object for parsing all Residues from ATOM lines and assign RSA (with biopython, DSSP or MuSiC) and distance matrix.
            * Parse list of all its amino acid residues
            * Manages non-standard amino acids
            * Ignore following models if there are more than 1
            * Computes RSA for each residue
            * Manages distance matrix of all pairs of residues in target chains.

        arguments:
            pdb_path (str):                              path to PDB file
            target_chains (str):                         target chains in the PDB (for distance matrix)
            homomeric_chains (str=None)                  homomer groups to consider inter-chains contacts (e.g. 'ABC' for homo-trimer or 'AC:BD' for two homo-dimers)
            ignore_hydrogen_atoms (bool=True):           set True to ignore hydrogen atoms in distance matrix evaluation
            ignore_backbone_atoms (bool=True):           set True to ignore backbone atoms in distance matrix evaluation
            distance_cache_path (Union[None, str]=None): path to write/read to/from distance matrix values (should be a '.npy' file)
            solve_rsa (bool=False):                      Solve RSA for residues
            rsa_cache_path (Union[None, str]=None):      path to write/read to/from RSA values
            logger (bool=False):                         set True for logs

        usage:
            structure = Structure("./my_pdb.pdb", "A")
            res_arr = structure.residues
            d = structure.distance_matrix
        """

        # Guardians
        if not isinstance(pdb_path, Sequence):
            if not os.path.isfile(pdb_path):
                raise FileNotFoundError(f"ERROR in Structure(): pdb_path='{pdb_path}' file does not exists.")
            if not StructureReader.has_valid_extention(pdb_path):
                raise ValueError(f"ERROR in Structure(): pdb_path='{pdb_path}' should have extention among {StructureReader.ACCEPTED_EXTENTIONS}.")
            if len(target_chains) != len(set(target_chains)):
                raise ValueError(f"ERROR in Structure(): target_chains='{target_chains}' can not contain any repeating characters.")
            for chain in target_chains:
                if chain not in self.ACCEPTED_CHAIN_DESCRIPTORS:
                    raise ValueError(
                        f"ERROR in Structure(): target_chains='{target_chains}' contain forbidden character '{chain}' "
                        f" -> allowed characters: '{self.ACCEPTED_CHAIN_DESCRIPTORS}'."
                    )

        # Set base properties
        if isinstance(pdb_path, Sequence): # Case: create fully-connected decoy Structure
            self.pdb_path = f"{pdb_path.name}_full.pdb"
            self.pdb_name = f"{pdb_path.name}_full"
        else:
            self.pdb_path = str(pdb_path)
            self.pdb_name = StructureReader.get_pdb_name(self.pdb_path)
        self.target_chains = target_chains
        self.name = f"{self.pdb_name}_{self.target_chains}"
        self.ignore_hydrogen_atoms = bool(ignore_hydrogen_atoms)
        self.ignore_backbone_atoms = bool(ignore_backbone_atoms)
        self.distance_cache_path = distance_cache_path
        self.solve_rsa = bool(solve_rsa)
        self.solve_distances = bool(solve_distances)
        self.rsa_cache_path = rsa_cache_path
        self.homomeric_chains = homomeric_chains

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger, disable_warnings=not logger)

        # Case of generating a decoy fully connected structure
        if isinstance(pdb_path, Sequence):
            self.fully_connected(template_sequence=pdb_path)
            return

        # Parse Structure
        self.logger.step(f"Parse 3D structure '{self.pdb_name}' (target chains '{target_chains}').")
        self.all_chains: str = ""
        self.residues: List[Residue] = []
        self.target_residues: List[Residue] = []
        self.residues_map: Dict[str, Residue] = {}
        self.expdta_line = None
        self._parse_structure()

        # Map atoms between homomeric chains
        self._map_homomeric_chains()

        # Compute distance matrix
        self.logger.step(f"Compute distances.")
        self.distance_matrix: NDArray[np.float32] = np.zeros([0, 0], dtype=np.float32)
        if self.solve_distances:
            self._compute_distance_matrix()

        # plDDT sanity check: if all plDDTs are 0, they are all set to 100
        self._sanitize_plddt()

    def fully_connected(
            self,
            template_sequence: Sequence,
        ) -> None:
        """
        Set Structure as single-chain Structure of sequence template_sequence.
            - distance_matrix are all 0.0 (fully connected structure)
            - RSA are all 0.0 (only core residues)
            - pLDDT are all 100.0 (perfectly resolved residues)
        """

        # Guardians
        assert len(template_sequence) > 1, f"ERROR in Structure().fully_connected(): template_sequence={template_sequence} should be non-empty."
        assert len(self.target_chains) == 1, f"ERROR in Structure().fully_connected(): target_chains='{self.target_chains}' should be of length 1."

        # Log
        self.logger.step(f"Generate decoy fully-connected structure '{self.pdb_name}' (target chains '{self.target_chains}').")

        # Init base properties
        chain = self.target_chains
        self.all_chains = chain
        self.expdta_line = None

        # Init residues
        self.residues = [
            Residue(chain, str(i+1), AminoAcid(aa_one), coords=np.zeros(3, dtype=np.float32), rsa=0.0, plddt=100.0)
            for i, aa_one in enumerate(template_sequence.sequence)
        ]
        self.target_residues = self.residues
        self.residues_map = {res.resid: res for res in self.residues}

        # Init distance matrix
        L = len(template_sequence)
        self.distance_matrix = np.zeros((L, L), dtype=np.float32)

    def _parse_structure(self) -> None:
        """Parse residues data from 3D structure file."""

        # Parse structure with biopython
        self.logger.log(f" * parse 3D structure file")
        bp_structure: BPStructure = StructureReader.read_biopython_structure(self.pdb_path)

        # Assign EXPDTA (experimental method)
        bp_header: dict
        try:
            bp_header = bp_structure.header
        except:
            bp_header = {}
        self.expdta_line = bp_header.get("structure_method", None)
        if isinstance(self.expdta_line, str):
            self.expdta_line = self.expdta_line.upper()

        # Manage multiple models: consider only model 1
        bp_model_0: BPModel = bp_structure[0]
        if len(bp_structure) > 1:
            self.logger.warning(f"3D structure contains multiple models ({len(bp_structure)}), but only model 0 will be considered.")

        # Remove hydrogen atoms if required
        # -> do before evaluating RSA for consistency between structures with and without hydrogen atoms
        # -> for example X-ray 3D structures has no hydrogen atoms but some AlphaFold models do
        if self.ignore_hydrogen_atoms:
            for bp_chain in bp_model_0:
                for bp_residue in bp_chain:
                    atoms_to_remove = []
                    for bp_atom in bp_residue:
                        atom_id = bp_atom.id
                        if any([atom_id.startswith(hp) for hp in self.HYDROGEN_ATOMS_PREFIXES]):
                            atoms_to_remove.append(atom_id)
                    for atom_id in atoms_to_remove:
                        bp_residue.detach_child(atom_id)

        # Compute SASA with biopython
        rsa_map = {}
        if self.solve_rsa:
            if self.rsa_cache_path is not None and os.path.isfile(self.rsa_cache_path):
                self.logger.log(f" * read RSA values from rsa_cache_path '{self.rsa_cache_path}'")
                rsa_map = read_rsa_map(self.rsa_cache_path)
                for bp_chain in bp_model_0: # guarantee residue.sasa property to avoid eventual bugs
                    for bp_residue in bp_chain:
                        bp_residue.sasa = None
            else:
                rsa_map = None # set rsa_map to None so that RSA is taken from Biopython ShrakeRupley
                self.logger.log(f" * solve RSA values using Shrake & Rupley algorithm")
                ShrakeRupley().compute(bp_model_0, level="R")

        # Extract residues information
        self.logger.log(f" * process structure object")
        bp_residue: BPResidue
        n_residues_failed_to_parse = 0
        n_residues_without_coords = 0
        n_residues_total = 0
        warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.Polypeptide")
        for bp_chain in bp_model_0:
            peptides = PPBuilder().build_peptides(bp_chain, aa_only=0) # use PPBuilder to keep only protein chains and exclude ligands
            for peptide in peptides:
                for bp_residue in peptide:

                    # Parse residue
                    n_residues_total += 1
                    try:
                        residue = Residue.from_biopython_residue(
                            bp_residue,
                            ignore_backbone_atoms=self.ignore_backbone_atoms,
                            rsa_map=rsa_map,
                        )
                    except:
                        n_residues_failed_to_parse += 1
                        continue

                    # Set residue
                    if residue.coords.size == 0:
                        n_residues_without_coords += 1
                        continue
                    self.residues.append(residue)
                    self.residues_map[residue.resid] = residue

        # Residues parsing warnings
        if n_residues_failed_to_parse > 0:
            self.logger.warning(
                f"failed to parse some residues from structure:"
                f" {n_residues_failed_to_parse} / {n_residues_total}"
            )
        if n_residues_without_coords > 0:
            self.logger.warning(
                f"some residues from structure were ignored because they have no atoms usable by StructureDCA:"
                f" {n_residues_without_coords} / {n_residues_total}"
            )

        # Set all_chains (non-redundent string of all chains)
        self.all_chains = "".join(dict.fromkeys([res.chain for res in self.residues]))
        for chain in self.target_chains:
            if chain not in self.all_chains:
                raise ValueError(f"ERROR in {self}: target chain '{chain}' not found among chains contained in the 3D structure ('{self.all_chains}').")

        # Order residues + set target_residues
        # -> order: (1) residues from target_chains; (2) other residues
        residues_ordered: List[Residue] = []
        for chain in self.target_chains:
            for residue in self.residues:
                if residue.chain == chain:
                    residues_ordered.append(residue)
                    self.target_residues.append(residue)
        for residue in self.residues:
            if residue.chain not in self.target_chains:
                residues_ordered.append(residue)
        self.residues = residues_ordered

        # Order chains
        all_chains_ordered = self.target_chains
        for chain in self.all_chains:
            if chain not in self.target_chains:
                all_chains_ordered += chain
        self.all_chains = all_chains_ordered

        # Check RSA coherence
        if self.solve_rsa:
            self._verify_rsa_values()

        # Write RSA cache
        if self.rsa_cache_path is not None and not os.path.isfile(self.rsa_cache_path):
            self.logger.log(f" * save RSA values to rsa_cache_path '{self.rsa_cache_path}'")
            rsa_map = {res.resid: res.rsa for res in self.residues if res.rsa is not None}
            write_rsa_map(self.rsa_cache_path, rsa_map)

        # Log
        self.logger.log(f" * target chains: '{self.target_chains}' (l={len(self.target_residues)})")
        self.logger.log(f" * all chains:    '{self.all_chains}' (l={len(self.residues)})")

    def _map_homomeric_chains(self) -> None:
        """Map atoms coordinates between PDB chain(s) that are the same protein as the target chain(s)
            - such that we consider inter-chains interactions to evaluate distances between residues
            - e.g. use 'ABC' for trimer or 'AC:BD' for two dimers
        """

        # Base case: nothing to map
        if self.homomeric_chains is None:
            return None

        # Guardians: validate homomeric_chains object
        if not isinstance(self.homomeric_chains, str):
            msg = (
                f"ERROR in StructureDCA.structure._map_homomeric_chains(homomeric_chains='{self.homomeric_chains}'):\n"
                f" -> homomeric_chains must be a string."
            )
            raise ValueError(msg)
        
        homomeric_groups: List[str] = self.homomeric_chains.split(":")
        for homemer_group in homomeric_groups:
            if len(homemer_group) == 0:
                msg = (
                    f"ERROR in StructureDCA.structure._map_homomeric_chains(homomeric_chains='{self.homomeric_chains}'):\n"
                    f" -> homomeric_chains can not contain any empty homomer group."
                )
                raise ValueError(msg)
            n_target_chains_in_group = len([chain for chain in homemer_group if chain in self.target_chains])
            if n_target_chains_in_group > 1:
                msg = (
                    f"ERROR in StructureDCA.structure._map_homomeric_chains(homomeric_chains='{self.homomeric_chains}'):\n"
                    f" -> homemer_group '{homemer_group}' can not contain multiple target chains."
                )
                raise ValueError(msg)
        
        all_chains = "".join(homomeric_groups)
        if len(all_chains) != len(set(all_chains)):
            msg = (
                f"ERROR in StructureDCA.structure._map_homomeric_chains(homomeric_chains='{self.homomeric_chains}'):\n"
                f" -> homomeric_chains can not contain repeated chains."
            )
            raise ValueError(msg)
        for chain in all_chains:
            if chain not in self.all_chains:
                msg = (
                    f"ERROR in StructureDCA.structure._map_homomeric_chains(homomeric_chains='{self.homomeric_chains}'):\n"
                    f" -> chain '{chain}' is not found in Structure (among '{self.all_chains}')."
                )
                raise ValueError(msg)
        
        # Log
        self.logger.step(f"Map atoms from homomeric chain(s) '{self.homomeric_chains}' (for distance matrix evaluation).")

        # Construct homomeric chains mapping {}
        homomeric_chains_map: Dict[str, str] = {}
        for homemer_group in homomeric_groups:
            target_chains = [chain for chain in homemer_group if chain in self.target_chains]
            template_chains = [chain for chain in homemer_group if chain not in self.target_chains]
            if len(target_chains) == 0:
                continue
            for template_chain in template_chains:
                homomeric_chains_map[template_chain] = target_chains[0]

        # Loop on chain mapping
        for template_chain, target_chain in homomeric_chains_map.items():

            # Get residues
            template_residues = [res for res in self.residues if res.chain == template_chain]
            target_residues = [res for res in self.residues if res.chain == target_chain]
            template_sequences = Sequence(f"{self.name}_{template_chain}", "".join([res.amino_acid.one for res in template_residues]))
            target_sequences = Sequence(f"{self.name}_{target_chain}", "".join([res.amino_acid.one for res in target_residues]))

            # Align the two homomeric chains and get resid mapping
            homomeric_chains_alignment = PairwiseAlignment(target_sequences, template_sequences)
            homomeric_residues_map = homomeric_chains_alignment.get_mapping(
                ids1=[res.resid for res in target_residues],
                ids2=[res.resid for res in template_residues],
            )

            # Add atoms from template chain to target chain
            n_assigned, n_not_in_template, n_mismatch = 0, 0, 0
            for target_residue in target_residues:
                target_resid = target_residue.resid
                if target_resid not in homomeric_residues_map:
                    n_not_in_template += 1
                    continue
                template_resid = homomeric_residues_map[target_resid]
                template_residue = self.get_residue(template_resid)
                if target_residue.amino_acid.one != template_residue.amino_acid.one:
                    n_mismatch += 1
                    continue
                target_residue.coords = np.concatenate([target_residue.coords, template_residue.coords])
                n_assigned += 1

            # Log warnings
            if n_not_in_template > 0:
                self.logger.warning(f"{n_not_in_template} / {len(target_residues)}: residue(s) in target chain '{target_chain}' with not corresponding residue in homomeric chain '{template_chain}'.")
            if n_mismatch > 0:
                self.logger.warning(f"{n_mismatch} / {len(target_residues)}: residue(s) in target chain '{target_chain}' with mismatching residue in homomeric chain '{template_chain}'.")

            # Log
            self.logger.log(f" * map atoms from chain '{template_chain}' ({len(template_residues)}) to chain '{target_chain}' ({len(target_residues)}): {n_assigned} / {len(target_residues)} target residues with new atom coordinates")

    def _compute_distance_matrix(self) -> None:
        """Compute all-atoms distance matrix between all pairs of residues in target chain."""

        # Log
        L = len(self.target_residues)
        self.logger.log(f" * compute residue-residue distance matrix [{L}x{L}]")

        # Load from cached file
        if self.distance_cache_path is not None and os.path.isfile(self.distance_cache_path):
            self.logger.log(f" * read cached distance values from distance_cache_path: '{self.distance_cache_path}'")
            self.load_distance_matrix(self.distance_cache_path)

        # Compute new distance matrix
        else:
            distance_matrix = np.zeros((L, L), dtype=np.float32)
            for i1, res1 in enumerate(self.target_residues):
                for i2 in range(i1):
                    res2 = self.target_residues[i2]
                    dist = self.distance_residues(res1, res2)
                    distance_matrix[i1, i2] = dist
                    distance_matrix[i2, i1] = dist
            self.distance_matrix = distance_matrix
            # Save if required
            if self.distance_cache_path is not None:
                self.logger.log(f" * save distance values to cache_path: '{self.distance_cache_path}'")
                self.save_distance_matrix(self.distance_cache_path)

    def _sanitize_plddt(self) -> None:
        """If pLDDT of all residues is 0 (meaning pLDDT is missing), they are all set to 100."""
        
        plddt_is_set = False

        # check if plDDT is != 0 for any residue
        for residue in self.residues:
            if residue.plddt != 0.0:
                plddt_is_set = True
                break
        
        # if not, raise warning and set all plDDTs to 100
        if not plddt_is_set:
            self.logger.warning("All residues in PDB file have plDDT=0. They were all set to 100.")
            for residue in self.residues:
                residue.plddt = 100.0


	# Base Properties ----------------------------------------------------------
    def __str__(self) -> str:
        return f"Structure('{self.name}', len={len(self.target_residues)})"

    def __contains__(self, resid: str) -> bool:
        return resid in self.residues_map

    def get_residue(self, resid: str) -> Residue:
        return self.residues_map[resid]


    # Get Methods --------------------------------------------------------------   
    def get_sequence(self, chain: str) -> Sequence:
        """Return Sequence of a given chain."""
        assert len(chain) == 1, f"ERROR in {self}.get_sequence(): chain='{chain}' should be a length 1 string chain descriptor."
        assert chain in self.all_chains, f"ERROR in {self}.get_sequence(): chain='{chain}' is not among detected protein chains {self.all_chains}."
        seq = "".join(res.amino_acid.one for res in self.residues if res.chain == chain)
        return Sequence(f"{self.pdb_name}_{chain}", seq)
    
    def get_target_sequence(self) -> Sequence:
        seq = "".join(res.amino_acid.one for res in self.target_residues)
        return Sequence(self.name, seq)
    
    def get_fasta(self, chains: Union[None, str]=None) -> str:
        """Get fasta string of a given chain (or of all chains by default)."""
        if chains is None:
            chains = self.all_chains
        fasta_str_list = [self.get_sequence(c).to_fasta_string() for c in chains]
        return "".join(fasta_str_list)
    
    def write_fasta(self, fasta_path: str, chains: Union[None, str]=None) -> str:
        """Write fasta string of a given chain (or of all chains by default)."""
        assert fasta_path.endswith(".fasta"), f"ERROR in {self}.write_fasta(): fasta_path='{fasta_path}' should end with '.fasta'."

        # Create output directory if it does not exist
        os.makedirs(os.path.dirname(fasta_path), exist_ok=True)

        fasta_str = self.get_fasta(chains)
        with open(fasta_path, "w") as fs:
            fs.write(fasta_str)
        return fasta_str

    def is_experimental(self) -> bool:
        """Return True is the Structure is detected as experimental by its 'EXPDTA' line."""
        EXPERIMENTAL_METHODS = [
            'X-RAY DIFFRACTION', 'NEUTRON DIFFRACTION', 'FIBER DIFFRACTION',
            'SOLUTION NMR', 'SOLID-STATE NMR', 'ELECTRON MICROSCOPY', 'ELECTRON CRYSTALLOGRAPHY',
            'FLUORESCENCE TRANSFER', 'INFRARED SPECTROSCOPY', 'SOLUTION SCATTERING', 'EPR',
        ]
        if self.expdta_line is None:
            return False
        return any([method in self.expdta_line for method in EXPERIMENTAL_METHODS])
    
    def is_probably_experimental(self) -> bool:
        """Return True is the Structure is detected as probably experimental by its RSA vs. B-factor correlation."""

        # Case: Structure is experimental
        if self.is_experimental():
            return True
        rsa_arr = np.array([res.rsa for res in self.residues])
        plddt_arr = np.array([res.plddt for res in self.residues])

        # Case: RSA or pLDDT array is constant
        if np.all(rsa_arr == rsa_arr[0]):
            return False
        if np.all(plddt_arr == plddt_arr[0]):
            return False
        
        # Base case: rely on positive correlation between B-factor (in place of pLDDT) and RSA
        corr = np.corrcoef(rsa_arr, plddt_arr)[0, 1]
        return corr > 0.0


    # Distances ----------------------------------------------------------------
    @staticmethod
    def distance_residues(res1: Residue, res2: Residue) -> np.float32:
        """Return distance between two residues."""
        diff_matrix = res1.coords[:, np.newaxis, :] - res2.coords[np.newaxis, :, :]
        return np.sqrt(np.min(np.sum(diff_matrix ** 2, axis=2)))
    
    def save_distance_matrix(self, matrix_path: str) -> "Structure":
        """Save distance matrix to a '.npy' file."""
        assert matrix_path.endswith(".npy"), f"ERROR in {self}.save_distance_matrix(): matrix_path='{matrix_path}' should be a '.npy' file."
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        np.save(matrix_path, self.distance_matrix)
        return self
    
    def read_distance_matrix(self, matrix_path: str) -> NDArray[np.float32]:
        """Read and return distance matrix from a '.npy' file."""
        assert matrix_path.endswith(".npy"), f"ERROR in {self}.read_distance_matrix(): matrix_path='{matrix_path}' should be a '.npy' file."
        assert os.path.isfile(matrix_path), f"ERROR in {self}.read_distance_matrix(): matrix_path='{matrix_path}' file does not exists."
        return np.load(matrix_path)

    def load_distance_matrix(self, matrix_path: str) -> "Structure":
        """Load distance matrix from a '.npy' file."""
        distance_matrix = self.read_distance_matrix(matrix_path)
        L = len(self.target_residues)
        self.distance_matrix = distance_matrix
        assert self.distance_matrix.shape == (L, L), f"ERROR in {self}.load_distance_matrix(): matrix shape {self.distance_matrix.shape} does not match length {L} of target chains {self.target_chains}."
        return self


    # Dependencies -------------------------------------------------------------
    def _verify_rsa_values(self) -> None:
        """Warnings and Errors for non-assigned RSA residues."""

        # No RSA errors
        if all(res.rsa is None for res in self.residues):
            raise ValueError(f"ERROR in Structure(): zero RSA values for 3D structure '{self.pdb_path}'.")
        if all(res.rsa is None for res in self.target_residues):
            raise ValueError(f"ERROR in Structure(): zero RSA values for target chains '{self.target_chains}' of 3D structure '{self.pdb_path}'.")

        # Log RSA state
        n_assigned_target_chains = sum(res.rsa is not None for res in self.target_residues)
        self.logger.log(f" * assigned RSA values: {n_assigned_target_chains} / {len(self.target_residues)} ")

        # Some missing RSA warnings
        norsa_std, norsa_non_std = 0, 0
        for residue in self.target_residues:
            if residue.rsa is None:
                if residue.amino_acid.is_standard():
                    norsa_std += 1
                else:
                    norsa_non_std += 1
        norsa = norsa_std + norsa_non_std
        if norsa > 0:
            self.logger.warning(
                f"{norsa} / {len(self.target_residues)} residues with no assigned RSA values "
                f"({norsa_std} std and {norsa_non_std} non-std) in PDB target chains '{self.target_chains}'."
                "\n   -> This can be caused by non-standard AAs or missing atoms."
                "\n   -> For optimal RSA estimations, we recommend to 'repair' the PDB and standardize AAs."
            )
