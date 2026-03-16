
# Imports ----------------------------------------------------------------------
from typing import Dict, Union, List
import numpy as np
from numpy.typing import NDArray
from structuredca.utils import Logger
from structuredca.structure import Structure, Residue
from structuredca.sequence import Sequence, PairwiseAlignment
from structuredca.aligner import LinearExtrapolation

# Main -------------------------------------------------------------------------
class StructureSequenceAlignment:
    """
    Class to merge (sometimes slightly diverging) information from Structure (PDB) to Sequence (Fasta).
        * Extrapolates and maps distance_matrix derived from structure to a corresponding Sequence.
        * Manages mapping between residues / mutations from Structure (as referenced in the PDB) to Sequence (as referenced in the Fasta) and conversly.

    usage:
        str_seq_align = StructureSequenceAlignment(my_structure: ProteinStructure, my_sequence: Sequence)
        dist_matrix_sequence = str_seq_align.distance_matrix
        res_position_str = str_seq_align.seq_2_str_residues['13']
        res_position_seq = str_seq_align.str_2_seq_residues['A4']
    """

    # Constants ----------------------------------------------------------------

    # Extrapolated distance K for missing residues in structure between two adjacent residues
    #    -> d[i, i+1] = K when residue i is missing
    DISTANCE_ADJ_BACKBONE = 1.34
    DISTANCE_ADJ_NOBACKBONE = 5.70

    # Increment K to extrapolate distances of missing residues in structure for other residues
    #    -> d[i, j] = min(d[i+1, j], d[i+1, j]) + K when residue i is missing
    DISTANCE_INCREMENT_BACKBONE = 3.40
    DISTANCE_INCREMENT_NOBACKBONE = 5.00

    # Increment K to extrapolate RSA of missing tail residues
    #    -> rsa[p+i] = rsa[p] + i*K when tail residues (p+1, p+2, ...) are missing
    RSA_TAIL_INCREMENT = 20.0


    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            structure: Structure,
            sequence: Sequence,
            logger: Union[bool, Logger]=False,
        ):

        # Init base properties and alignment -----------------------------------

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)
        self.logger.step(f"Align structure (PDB, l={len(structure.target_residues)}) and sequence (MSA, l={len(sequence)}).")

        # Init base properties
        self.target_chains = structure.target_chains
        self.seq_structure : Sequence = structure.get_target_sequence()
        self.seq_sequence: Sequence = sequence
        if structure.ignore_backbone_atoms:
            self.distance_increment: float = self.DISTANCE_INCREMENT_NOBACKBONE
            self.distance_adj: float = self.DISTANCE_ADJ_NOBACKBONE
        else:
            self.distance_increment: float = self.DISTANCE_INCREMENT_BACKBONE
            self.distance_adj: float = self.DISTANCE_ADJ_BACKBONE

        # Alignment
        self.align = PairwiseAlignment(self.seq_structure, self.seq_sequence, query_insertion_multiplier=3.0)
        self.align_structure: str = self.align.align1
        self.align_sequence: str = self.align.align2

        # Residues mapping
        res_positions_structure = [res.resid for res in structure.target_residues]
        res_positions_sequence = [str(i) for i in range(1, len(self.seq_sequence) + 1)]
        self.str_2_seq_residues: Dict[str, str] = self.align.get_mapping(res_positions_structure, res_positions_sequence)
        self.seq_2_str_residues: Dict[str, str] = {v: k for k, v in self.str_2_seq_residues.items()}
        
        # List of structural residues aligned to MSA positions
        self.residues_array: List[Union[None, Residue]] = []
        for resid_seq in res_positions_sequence:
            resid_str = self.seq_2_str_residues.get(resid_seq)
            if resid_str is None:
                self.residues_array.append(None)
                continue
            self.residues_array.append(structure.get_residue(resid_str))

        # Log
        n_positions_with_structure = self.align.match + self.align.mismatch
        self.logger.log(f" * sequence positions aligned to structure: {n_positions_with_structure} / {len(self.seq_sequence)}")

        # Extrapolate distance matrix to sequence ------------------------------

        # Log
        if n_positions_with_structure != len(self.seq_sequence):
            self.logger.log(" * extrapolate distance matrix to positions without structural aligment")

        # Init extrapolation
        extrapolation = LinearExtrapolation(self.distance_increment, self.distance_adj)
        self.distance_matrix: NDArray[np.float32] = structure.distance_matrix
        self.rsa_array: NDArray[np.float32] = np.zeros([0], dtype=np.float32)
        self.plddt_array: NDArray[np.float32] = np.zeros([0], dtype=np.float32)

        # Extrapolate Left Tail Gaps
        n_left_tail_str_gaps = self.align.left_gap1
        if n_left_tail_str_gaps > 0:
            first_residue_dist_arr = self.distance_matrix[0,:]
            N = n_left_tail_str_gaps
            diag_matrix = extrapolation.extrapolate_diagonal_matrix(N)
            marg_matrix = extrapolation.extrapolate_marginal_matrix(first_residue_dist_arr, N, reverse_lines=True)
            self.distance_matrix = np.concatenate(
                (
                    np.concatenate((diag_matrix, marg_matrix), axis=1),
                    np.concatenate((marg_matrix.T, self.distance_matrix), axis=1)
                ),
                axis=0,
            )

        # Extrapolate Internal Gaps
        n_internal_str_gaps = 0
        internal_gaps_range_list = PairwiseAlignment.get_gaps_ranges(self.align_structure, tail_gaps=False)
        for gap_range in internal_gaps_range_list:
            i1, i2 = gap_range
            N = i2 - i1
            n_internal_str_gaps += N

            # Generate Interpolated values
            left_residue_dist_arr = self.distance_matrix[i1-1, :]
            right_residue_dist_arr = self.distance_matrix[i1, :]
            I_diag = extrapolation.extrapolate_diagonal_matrix(N) # Extrapolated distances between missing residues
            I_left = extrapolation.extrapolate_marginal_matrix(left_residue_dist_arr, N, reverse_lines=False)
            I_right = extrapolation.extrapolate_marginal_matrix(right_residue_dist_arr, N, reverse_lines=True)
            I = np.minimum(I_left, I_right) # Consensus distance extrapolated from left and right residue distances
            I1, I2 = I[:, :i1], I[:, i1:] # Split I in parts
            
            # Split existing distance matrix in 4 submatrices
            D11 = self.distance_matrix[:i1, :i1]
            D12 = self.distance_matrix[:i1, i1:]
            D21 = self.distance_matrix[i1:, :i1]
            D22 = self.distance_matrix[i1:, i1:]

            # Recompose new matrix with D and I
            self.distance_matrix = np.concatenate(
                (
                    np.concatenate((D11, I1.T, D12), axis=1),
                    np.concatenate((I1, I_diag, I2), axis=1),
                    np.concatenate((D21, I2.T, D22), axis=1),
                ),
                axis=0,
            )

        # Extrapolate Right Tail Gaps
        n_right_tail_str_gaps = self.align.right_gap1
        if n_right_tail_str_gaps > 0:
            last_residue_dist_arr = self.distance_matrix[-1,:]
            N = n_right_tail_str_gaps
            diag_matrix = extrapolation.extrapolate_diagonal_matrix(N)
            marg_matrix = extrapolation.extrapolate_marginal_matrix(last_residue_dist_arr, N, reverse_lines=False)
            self.distance_matrix = np.concatenate(
                (
                    np.concatenate((self.distance_matrix, marg_matrix.T), axis=1),
                    np.concatenate((marg_matrix, diag_matrix), axis=1)
                ),
                axis=0,
            )

        # Extrapolate RSA ------------------------------------------------------
        residues_arr = structure.target_residues
        if structure.solve_rsa:

            # Init RSA array (mapped to Structure-Sequence alignment)
            rsa_arr = []
            missing_rsa_arr = [] # Alignment string with gap '-' when RSA is missing
            res_i = 0
            n_missing_rsa_values = 0
            for char in self.align.align1:
                # Case: residue not in structure
                if char == "-":
                    rsa_arr.append(None)
                    missing_rsa_arr.append("-")
                    continue
                residue = residues_arr[res_i]
                res_rsa = residue.rsa
                res_i += 1
                # Case: residue in structure but no RSA assigned to it
                if res_rsa is None:
                    rsa_arr.append(None)
                    missing_rsa_arr.append("-")
                    n_missing_rsa_values += 1
                # Case: residue in structure with assigned RSA
                else:
                    rsa_arr.append(res_rsa)
                    missing_rsa_arr.append(residue.amino_acid.one)
            missing_rsa_str = "".join(missing_rsa_arr)

            # Complete missing RSA in structure
            missing_rsa_ranges = PairwiseAlignment.get_gaps_ranges(missing_rsa_str, tail_gaps=True) # Intervals of missing RSA values

            # Log
            if len(missing_rsa_ranges) > 0:
                self.logger.log(" * extrapolate RSA to positions without structural aligment")
            for missing_rsa_range in missing_rsa_ranges:
                
                # Init
                i1, i2 = missing_rsa_range
                l = i2 - i1
                
                # Case of Left residues missing RSA
                if i1 == 0:
                    next_rsa = rsa_arr[i2]
                    for i in range(1, l + 1): # Assign as next residue RSA with an increment to increasing RSA values
                        rsa_arr[i2-i] = min(next_rsa + i*self.RSA_TAIL_INCREMENT, 100.0)

                # Case of Rights residues missing RSA
                elif i2 == len(rsa_arr):
                    previous_rsa = rsa_arr[i1-1]
                    for i in range(1, l+1): # Assign as previous residue RSA with an increment to increasing RSA values
                        rsa_arr[i1-1+i] = min(previous_rsa + i*self.RSA_TAIL_INCREMENT, 100.0)

                # Case of interior residues missing RSA
                else:
                    next_rsa = rsa_arr[i2]
                    previous_rsa = rsa_arr[i1-1]
                    for i in range(1, l+1): # Assign as average next and previous residues RSA
                        rsa_arr[i1-1+i] = np.mean([next_rsa, previous_rsa])

            self.rsa_array = np.array(rsa_arr, dtype=np.float32)

        # Extrapolate pLDDT (just set zeros if missing data) -------------------

        # Init pLDDT array (mapped to Structure-Sequence alignment)
        plddt_array = []
        default_plddt = 0.0
        res_i = 0
        n_missing_plddt_values = 0
        for char in self.align.align1:
            # Case: residue not in structure
            if char == "-":
                plddt_array.append(default_plddt)
                n_missing_plddt_values += 1
                continue
            residue = residues_arr[res_i]
            res_plddt = residue.plddt
            res_i += 1
            # Case: residue in structure but no pLDDT assigned to it
            if res_plddt is None:
                plddt_array.append(default_plddt)
                n_missing_plddt_values += 1
            # Case: residue in structure with assigned pLDDT
            else:
                plddt_array.append(res_plddt)
        self.plddt_array = np.array(plddt_array, dtype=np.float32)

        # Remove missing positions in sequence ---------------------------------
        # (from distance matrix and RSA array)
        if self.align.gap2 > 0:
            sequence_ids = np.array([i for i, aa in enumerate(self.align_sequence) if aa != "-"])
            self.distance_matrix = self.distance_matrix[sequence_ids, :] # Trim in dim 1
            self.distance_matrix = self.distance_matrix[:, sequence_ids] # Trim in dim 2
            if structure.solve_rsa:
                self.rsa_array = self.rsa_array[sequence_ids] # Trim array ids
            self.plddt_array = self.plddt_array[sequence_ids] # Trim array ids

        # Alignment Warnings ---------------------------------------------------
        if n_left_tail_str_gaps > 0:
            self.logger.warning(f"{n_left_tail_str_gaps} / {len(self)} missing left tail residues in PDB (some distances and RSA are extrapolated).")
        if n_right_tail_str_gaps > 0:
            self.logger.warning(f"{n_right_tail_str_gaps} / {len(self)} missing right tail residues in PDB (some distances and RSA are extrapolated).")
        if n_internal_str_gaps > 0:
            self.logger.warning(f"{n_internal_str_gaps} / {len(self)} missing internal residues in PDB (some distances and RSA are extrapolated).")
        if structure.solve_rsa:
            if n_missing_rsa_values > 0:
                self.logger.warning(f"{n_missing_rsa_values} / {len(self)} aligned residues from PDB without assigned RSA (some RSA are extrapolated).")
        if n_missing_plddt_values > 0:
            self.logger.warning(f"{n_missing_plddt_values} / {len(self)}: residues without a pLDDT score (thus extrapolated to zero).")
        critical_alignment_warning = False
        if self.align.mismatch > 0:
            critical_alignment_warning = True
            self.logger.warning(f"{self.align.mismatch} / {len(self)} mismatch between MSA and PDB.", critical=True)
        if self.align.internal_gap2 > 0:
            critical_alignment_warning = True
            self.logger.warning(f"{self.align.internal_gap2} internal residues in the PDB do not correspond to a position in MSA.", critical=True)
        if critical_alignment_warning and not self.logger.disable_warnings:
            self.align.show(n_lines=80, only_critical_chunks=True)


    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        return f"StructureSequenceAlignment(STR['{self.seq_structure.name}'] -> SEQ['{self.seq_sequence.name}'])"
    
    def __len__(self) -> int:
        return len(self.seq_sequence)
    
    def is_structure_residue_aligned(self, res_position: str) -> bool:
        """Return if a residue position (like '13') from a PDB is aligned to sequence."""
        return res_position in self.str_2_seq_residues 
    
    def is_sequence_residue_aligned(self, res_position: str) -> bool:
        """Return if a residue position (like '13') from a Fasta is aligned to structure."""
        return res_position in self.seq_2_str_residues
    
    def convert_to_sequence_mutation(self, mutations_str: str) -> str:
        """Convert a mutation on structure (PDB) reference to a mutation in sequence (Fasta) reference (Ex: 'MB13A' -> 'M4A')."""
        
        # Single mutation Case
        if ":" not in mutations_str:
            assert len(mutations_str) >= 4, f"ERROR in {self}.convert_to_sequence_mutation(): mutations_str='{mutations_str}' should be a string of length 4 or more."
            wt, chain, pos, mt = mutations_str[0], mutations_str[1], mutations_str[2:-1], mutations_str[-1]
            resid = chain + pos
            assert chain in self.target_chains, f"ERROR in {self}.convert_to_sequence_mutation(): mutations_str='{mutations_str}' chain is not StructureDCA target chains '{self.target_chains}'"
            assert resid in self.str_2_seq_residues, f"ERROR in {self}.convert_to_sequence_mutation(): mutations_str='{mutations_str}' residue position is not aligned."
            resid_translated = self.str_2_seq_residues[resid]
            return wt + resid_translated + mt
        
        # Multiple mutations Case
        return ":".join([self.convert_to_sequence_mutation(single_mut) for single_mut in mutations_str.split(":")])

    def convert_to_structure_mutation(self, mutations_str: str) -> str:
        """Convert a mutation on sequence (Fasta) reference to a mutation in structure (PDB) reference(Ex: 'M4A' -> 'MB13A')."""
        
        # Single mutation Case
        if ":" not in mutations_str:
            assert len(mutations_str) >= 3, f"ERROR in {self}.convert_to_structure_mutation(): mutations_str='{mutations_str}' should be a string of length 3 or more."
            wt, pos, mt = mutations_str[0], mutations_str[1:-1], mutations_str[-1]
            assert pos in self.seq_2_str_residues, f"ERROR in {self}.convert_to_structure_mutation(): mutations_str='{mutations_str}' residue position is not aligned."
            resid_translated = self.seq_2_str_residues[pos]
            return wt + resid_translated + mt
        
        # Multiple mutations Case
        return ":".join([self.convert_to_structure_mutation(single_mut) for single_mut in mutations_str.split(":")])

    def get_alignment_warning(self, i: int) -> Union[None, str]:
        """Return an alignment warning (str) if there is one for position [i] or None otherwise."""
        if i < 0 or i >= len(self.seq_sequence):
            return "out of bounds"
        aligned_residue = self.residues_array[i]
        if aligned_residue is None:
            return "missing structure"
        aa_sequence = self.seq_sequence.sequence[i]
        if aa_sequence != aligned_residue.amino_acid.one:
            return "mismatch structure"
        return None
    
    def get_alignment_warnings_list(self, list_i: List[int]) -> List[str]:
        """Return list of all different alignment warnings (str) at positions [list_i]."""
        warnings_list: List[str] = []
        for i in list_i:
            warning = self.get_alignment_warning(i)
            if warning is not None:
                warnings_list.append(warning)
        return list(dict.fromkeys(warnings_list))