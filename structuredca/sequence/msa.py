
# Imports ----------------------------------------------------------------------
import os.path
import numpy as np
from numpy.typing import NDArray
from typing import List, Union
from structuredca.utils import Logger
from structuredca.sequence import Sequence, FastaStream

# MSA --------------------------------------------------------------------------
class MSA:
    """
    Class to pre-process an MSA.
        * Remove redundent sequences
        * Upper case all AAs.
        * Remove twilight zone sequences if required
    """

    # Constants ----------------------------------------------------------------
    GAP_CHAR = "-"

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            msa_path: str,
            min_seqid: Union[None, float],
            logger: Union[bool, Logger],
        ):

        # Guardians and init
        assert os.path.isfile(msa_path), f"ERROR in StructureDCA::MSA(): msa_path='{msa_path}' file does not exists."
        self.msa_path = msa_path
        self.min_seqid = min_seqid
        self.sequences: List[Sequence] = []
        self.initial_depth = 0

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)

        # Read sequences from file
        fasta_stream = FastaStream(self.msa_path) # Caution with this one
        sequences_set = set()
        sequence = fasta_stream.get_next()
        if sequence is None:
            raise ValueError(f"ERROR in StructureDCA::MSA(): no sequences found in MSA '{msa_path}'.")
        tgt_seq_len = len(sequence)
        if tgt_seq_len == 0:
            raise ValueError(f"ERROR in StructureDCA::MSA(): target sequence of MSA '{msa_path}' is empty.")
        while sequence is not None:
            self._verify_sequence_length(sequence, tgt_seq_len, self.initial_depth)
            sequence_str = sequence.sequence
            if sequence_str not in sequences_set:
                self.sequences.append(sequence)
                sequences_set.add(sequence_str)
            sequence = fasta_stream.get_next()
            self.initial_depth += 1
        self.logger.log(f" * remove redundant sequences: {self.initial_depth} -> {len(self.sequences)}")
        fasta_stream.close()

        # Filter sequences that are too far from target sequence
        self._remove_far_seqid_sequences()

    # Base properties ----------------------------------------------------------
    def __str__(self) -> str:
        return f"MSA(l={self.length}, d={self.depth}, '{self.msa_path}')"

    @property
    def length(self) -> int:
        """Length of the target sequence of the MSA."""
        return len(self.target_sequence)
    
    @property
    def depth(self) -> int:
        """Number of sequences in the MSA (after pre-processing)."""
        return len(self.sequences)
    
    @property
    def target_sequence(self) -> Sequence:
        return self.sequences[0]

    # Methods ------------------------------------------------------------------    
    def gap_ratios_positions(self) -> NDArray[np.float32]:
        """Get array of gap ratios (for each position)."""
        GAP_CHAR = Sequence.GAP_CHAR
        gap_counts = np.zeros(self.length, dtype=int)
        for sequence in self.sequences:
            gap_indexes_in_sequence = [i for i, aa in enumerate(sequence) if aa == GAP_CHAR]
            gap_counts[gap_indexes_in_sequence] += 1
        return np.array(gap_counts / self.depth, dtype=np.float32)

    # IO Methods ---------------------------------------------------------------
    def write(self, msa_path: str) -> "MSA":
        """Save MSA to a FASTA MSA file."""

        # Guardians
        msa_path = os.path.abspath(msa_path)
        os.makedirs(os.path.dirname(msa_path), exist_ok=True)

        # Write
        with open(msa_path, "w") as fs:
            fs.write("".join([seq.to_fasta_string() for seq in self.sequences]))
        return self

    # Dependencies -------------------------------------------------------------
    def _verify_sequence_length(self, sequence: Sequence, target_length: int, i: int) -> None:
        """For coherence of all sequences in the MSA."""
        if len(sequence) != target_length:
            seq_str = sequence.sequence
            if len(seq_str) > 40:
                seq_str = seq_str[0:37] + "..."
            error_log = f"ERROR in StructureDCA::MSA(): msa_path='{self.msa_path}':"
            error_log += f"\n -> length of sequence [{i+1}] l={len(sequence)} ('{seq_str}') does not match length of target sequence l={target_length}."
            raise ValueError(error_log)
        
    def _remove_far_seqid_sequences(self) -> None:
        """Filter sequences that are too far from target sequence by sequence identity."""

        # No filter case
        if self.min_seqid is None:
            return None

        # Guardian
        assert 0.0 <= self.min_seqid < 1.0, f"ERROR in StructureDCA::MSA(): min_seqid={self.min_seqid} should be stricktly between 0 and 1."

        # Compute sequences to keep
        keep_sequences: List[Sequence] = []
        target_sequence_str = self.target_sequence.sequence
        for current_sequence in self.sequences:
            current_sequence_str = current_sequence.sequence

            # Compute seqid with target sequence
            current_seqid = self._seqid_to_target(target_sequence_str, current_sequence_str)

            if current_seqid > self.min_seqid:
                keep_sequences.append(current_sequence)

        # Update MSA sequences
        l1, l2 = len(self.sequences), len(keep_sequences)
        self.sequences = keep_sequences

        # Log results
        self.logger.log(f" * filter distant sequences (min_seqid={self.min_seqid:.2f}): {l1} -> {l2}")

        # Guardians
        if l2 == 0:
            error_log = f"ERROR in StructureDCA::MSA(): _remove_far_seqid_sequences(): no sequence left."
            error_log += f"\n - No sequences left in the MSA after removing sequences that are too far from target sequence (by sequence indentity)"
            error_log += f"\n - min_seqid={self.min_seqid}: please increase value or set to None."
            raise ValueError(error_log)
        
    def _seqid_to_target(self, seq1: str, seq2: str) -> float:
        """Computes sequence identity between two sequences in the MSA."""
        gap = self.GAP_CHAR
        num_identical_residues = sum([int(aa1 == aa2) for aa1, aa2 in zip(seq1, seq2)])
        num_aligned_residues = sum([int(aa != gap) for aa in seq2])
        return num_identical_residues / num_aligned_residues
