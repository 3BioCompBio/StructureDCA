
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Dict, Union
import numpy as np
from numpy.typing import NDArray
from structuredca.utils import Logger
from structuredca.sequence import AminoAcid
from structuredca.structure import Residue
from structuredca.sequence import Sequence, PairwiseAlignment
from structuredca.structure.rsa import RSABiopython


# Main -------------------------------------------------------------------------

class Structure:
    """Structure object for parsing all Residues from ATOM lines and assign RSA (with biopython, DSSP or MuSiC) and distance matrix."""


    # Constants ----------------------------------------------------------------
    ACCEPTED_CHAIN_DESCRIPTORS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


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
            assert os.path.isfile(pdb_path),  f"ERROR in Structure(): pdb_path='{pdb_path}' file does not exists."
            assert pdb_path.endswith(".pdb"), f"ERROR in Structure(): pdb_path='{pdb_path}' should be a '.pdb' file."
            assert len(target_chains) == len(set(target_chains)), f"ERROR in Structure(): target_chains='{target_chains}' can not contain any repeating characters."
            for chain in target_chains:
                assert chain in self.ACCEPTED_CHAIN_DESCRIPTORS, f"ERROR in Structure(): target_chains='{target_chains}' contain not allowed characted '{chain}' (allowed: '{self.ACCEPTED_CHAIN_DESCRIPTORS}')."

        # Set base properties
        if isinstance(pdb_path, Sequence): # Case: create fully-connected decoy Structure
            self.pdb_path = f"{pdb_path.name}_full.pdb"
            self.pdb_name = f"{pdb_path.name}_full"
        else:
            self.pdb_path = pdb_path
            self.pdb_name = os.path.basename(self.pdb_path).removesuffix(".pdb")
        self.target_chains = target_chains
        self.name = f"{self.pdb_name}_{self.target_chains}"
        self.ignore_hydrogen_atoms = ignore_hydrogen_atoms
        self.ignore_backbone_atoms = ignore_backbone_atoms
        self.distance_cache_path = distance_cache_path
        self.solve_rsa = solve_rsa
        self.rsa_cache_path = rsa_cache_path
        self.homomeric_chains = homomeric_chains

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)

        # Case of generating a decoy fully connected structure
        if isinstance(pdb_path, Sequence):
            self.fully_connected(template_sequence=pdb_path)
            return

        # Parse Structure
        self.logger.step(f"Parse PDB structure '{self.pdb_name}' (target chains '{target_chains}').")
        self.all_chains: str = ""
        self.residues: List[Residue] = []
        self.target_residues: List[Residue] = []
        self.residues_map: Dict[str, Residue] = {}
        self.expdta_line = None
        self._parse_structure()

        # Map atoms between homomeric chains
        self._map_homomeric_chains()

        # Compute distance matrix
        self.logger.step(f"Compute structural properties.")
        self.distance_matrix: NDArray[np.float32] = np.zeros([0, 0], dtype=np.float32)
        if solve_distances:
            self._compute_distance_matrix()

        # Solve RSA
        if solve_rsa:
            self._assign_rsa()

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
        """Parse residues data from PDB file."""

        # Parse sequence
        model_counter = 0
        current_chain = None
        opened_chains, closed_chains = set(), set()

        # Init backbone atoms
        BACKBONE_ATOMS = ["N", "H", "H1", "H2", "H3", "1H", "2H", "3H", "CA", "HA", "C", "O", "OXT"]
        GLY_BACKBONE_ATOMS = ["N", "H", "H1", "H2", "H3", "1H", "2H", "3H", "C", "O", "OXT"] # keep C-apha since GLY has no side-chains

        # Init hydrogen atoms prefixes
        HYDROGEN_PREFIXES = ["H", "1H", "2H", "3H"]

        # Parse PDB residues
        with open(self.pdb_path, "r", encoding="ISO-8859-1") as fs:
            line = fs.readline()
            while line:
                prefix = line[0:6]
                
                # Atom line
                if prefix == "ATOM  " or prefix == "HETATM":

                    # Ignore hydrogen atoms if required
                    atom_type = line[12:16].replace(" ", "")
                    if self.ignore_hydrogen_atoms and any([atom_type.startswith(hp) for hp in HYDROGEN_PREFIXES]):
                        line = fs.readline()
                        continue

                    # Ignore backbone atoms if required
                    aa_three = line[17:20]
                    if self.ignore_backbone_atoms:
                        if aa_three == "GLY":
                            if atom_type in GLY_BACKBONE_ATOMS:
                                line = fs.readline()
                                continue
                        else:
                            if atom_type in BACKBONE_ATOMS:
                                line = fs.readline()
                                continue

                    current_chain = line[21]
                    if current_chain in closed_chains: # discard ATOM line if chain is closed
                        line = fs.readline()
                        continue
                    position = line[22:26].replace(" ", "")
                    aa = AminoAcid.parse_three(aa_three)
                    if aa.is_unknown(): # discard non amino acid ATOM lines
                        line = fs.readline()
                        continue
                    if current_chain not in opened_chains:
                        opened_chains.add(current_chain)
                        self.all_chains += current_chain
                    resid = current_chain + position
                    coord = np.array([np.float32(line[30:38]), np.float32(line[38:46]), np.float32(line[46:54])], dtype=np.float32)
                    if resid in self.residues_map:
                        self.residues_map[resid].coords.append(coord)
                    else:
                        plddt = np.float32(line[60:66])
                        residue = Residue(current_chain, position, aa, coords=[coord], plddt=plddt)
                        self.residues.append(residue)
                        self.residues_map[resid] = residue
                
                # Manage multiple models: consider only model 1
                elif prefix == "MODEL ":
                    model_counter += 1
                    if model_counter > 1:
                        self.logger.warning(f"PDB contains multiple models, but only model 1 will be considered.")
                        break

                # Manage closed chains: ATOMS that appears after the chain is closed are not part of the protein chain
                elif prefix == "TER   " or prefix == "TER\n":
                    if current_chain is not None:
                        closed_chains.add(current_chain)

                # Find EXPDTA line (if it exists)
                elif prefix == "EXPDTA":
                    self.expdta_line = line

                line = fs.readline()

        # Verify target chains existance in PDB
        for chain in self.target_chains:
            assert chain in self.all_chains, f"ERROR in {self}: target chain '{chain}' not found among chains contained in the PDB ('{self.all_chains}')."

        # Set residues coordinates to numpy
        for residue in self.residues:
            residue.coords = np.array(residue.coords)

        # Order residues (by chain order from target_chains) and create target_residues
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
                target_residue.coords = np.concat([target_residue.coords, template_residue.coords])
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
        """ If plDDT of all residues is 0, they are all set to 100 """
        
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

    def _assign_rsa(self) -> None:
        """Assign RSA to residues of the Structure using an RSA Solver."""
        
        # Solve RSA
        solver = RSABiopython(verbose=self.logger.verbose)
        rsa_map = solver.run(self.pdb_path, rsa_cache_path=self.rsa_cache_path)

        # Fill RSA
        n_assigned_tot, n_assigned_target_chains = 0, 0
        for residue in self.residues:
            resid = residue.resid
            if resid in rsa_map:
                n_assigned_tot += 1
                if resid[0] in self.target_chains:
                    n_assigned_target_chains += 1
                residue.rsa = rsa_map[resid]

        # No RSA error
        if n_assigned_tot == 0:
            raise ValueError(f"ERROR in Structure(): {solver} gives zero RSA values for PDB '{self.pdb_path}'.")
        if n_assigned_target_chains == 0:
            raise ValueError(f"ERROR in Structure(): {solver} gives zero RSA values for target chains '{self.target_chains}' of PDB '{self.pdb_path}'.")

        # Log
        self.logger.log(f" * assigned RSA values: {n_assigned_target_chains} / {len(self.target_residues)} ")
        self._verify_rsa_values()


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

    def _verify_rsa_values(self) -> None:
        """Warnings for non-assigned RSA residues."""
        norsa_std, norsa_non_std = 0, 0
        for residue in self.target_residues:
            if residue.rsa is None:
                if residue.amino_acid.is_standard():
                    norsa_std += 1
                else:
                    norsa_non_std += 1
        norsa = norsa_std + norsa_non_std
        if norsa > 0:
            warning_log = f"{norsa} / {len(self.target_residues)} residues with no assigned RSA values ({norsa_std} std and {norsa_non_std} non-std) in PDB target chains '{self.target_chains}'."
            warning_log += "\n   -> This can be caused by non-standard AAs or missing atoms."
            warning_log += "\n   -> For optimal RSA estimations, we highly recommend to 'repair' the PDB and standardize AAs."
            self.logger.warning(warning_log)
