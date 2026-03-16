
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Tuple
from structuredca.sequence import AminoAcid
from structuredca.sequence import Mutation

# Sequence ---------------------------------------------------------------------
class Sequence:
    """Container class for an amino acid sequence (name, sequence and weight).
    
    usage:
        seq = Sequence('seq1', 'MQIFVKTLTGKTI--T') \n
        seq_name: str = seq.name \n
        seq_str: str = seq.sequence \n
        seq.write('./fasta/seq1.fasta')
    """

    # Constants ----------------------------------------------------------------
    HEADER_START_CHAR = ">"
    GAP_CHAR = AminoAcid.GAP_ONE
    ALL_AAS: List[AminoAcid] = AminoAcid.get_all()
    AMINO_ACIDS_IDENTITY_MAP = {aa.one: aa.one for aa in ALL_AAS} | {aa.one.lower(): aa.one.lower() for aa in ALL_AAS}

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            name: str,
            sequence: str,
            weight: float=1.0,
            to_upper: bool=True,
            remove_lower_case: bool=False,
            convert_special_characters: bool=True
        ):
        """Constructor for a (protein) Sequence object.
            name                         (str)         name of the sequence
            sequence                     (str)         amino acid sequence as a string
            weight                       (float=1.0)   weight of the sequence (in an MSA)
            to_upper                     (bool=True)   if True, convert all lower case amino acids to upper cases (such as in '.a2m' format)
            remove_lower_case            (bool=False)  if True, remove all lower case amino acids (such as in '.a3m' format to align sequences)
            convert_special_characters   (bool=True)   if True, convert all non-standard characters (like '.' or '_') to a gap '-' (such as in '.a2m' or '.a3m' format)
        """
        if to_upper and remove_lower_case:
            raise ValueError(f"ERROR in Sequence(): inconsistent settings to_upper=True and remove_lower_case=True (both can not be True at the same time).")
        if name.startswith(self.HEADER_START_CHAR):
            name = name.removeprefix(self.HEADER_START_CHAR)
        if to_upper:
            sequence = sequence.upper()
        if remove_lower_case:
            sequence = "".join(c for c in sequence if not c.islower())
        if convert_special_characters:
            gap = self.GAP_CHAR
            aa_map = self.AMINO_ACIDS_IDENTITY_MAP
            sequence = "".join([aa_map.get(aa, gap) for aa in sequence])
        self.name: str = name
        self.sequence: str = sequence
        self.weight: float = weight
    
    # Base properties ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequence)
    
    def seq_short(self, max_len: int=50) -> str:
        """Return first [max_len] amino acids of the sequence."""
        if len(self.sequence) > max_len:
            return f"{self.sequence[0:max_len]}..."
        return self.sequence

    def __str__(self) -> str:
        MAX_PRINT_LEN = 15
        seq_str = self.sequence
        if len(seq_str) > MAX_PRINT_LEN:
            seq_str = f"{seq_str[0:MAX_PRINT_LEN]}..."
        name_str = self.name
        if len(name_str) > MAX_PRINT_LEN:
            name_str = f"{name_str[0:MAX_PRINT_LEN]}..."
        return f"Sequence('{name_str}', seq='{seq_str}', l={len(self)})"
    
    def __eq__(self, other: "Sequence") -> bool:
        return self.sequence == other.sequence
    
    def __neq__(self, other: "Sequence") -> bool:
        return self.sequence != other.sequence
    
    def __hash__(self) -> int:
        return hash(self.sequence)
    
    def __iter__(self):
        return iter(self.sequence)
    
    def __getitem__(self, id: int) -> str:
        return self.sequence[id]
    
    def __contains__(self, char: str) -> bool:
        return char in self.sequence
    
    # Base Methods -------------------------------------------------------------
    def n_gaps(self) -> int:
        """Return number of gaps in sequence."""
        return len([char for char in self.sequence if char == self.GAP_CHAR])
    
    def n_non_gaps(self) -> int:
        """Return number of non-gaps in sequence."""
        return len([char for char in self.sequence if char != self.GAP_CHAR])
    
    def gap_ratio(self) -> float:
        """Return gap ratio."""
        return self.n_gaps() / len(self)

    def contains_gaps(self) -> bool:
        """Return is sequence contains gaps."""
        return self.GAP_CHAR in self.sequence
    
    def is_all_amino_acids(self) -> bool:
        """Returns is sequence is composed of only standard amino acids."""
        for char in self.sequence:
            if not AminoAcid.one_exists(char):
                return False
        return True
    
    def to_fasta_string(self) -> str:
        """Return string of the sequence in FASTA format."""
        return f"{self.HEADER_START_CHAR}{self.name}\n{self.sequence}\n"
    
    def is_mutation_compatible(self, mutation: Mutation) -> Tuple[bool, str]:
        """Verify if Mutation is compatible with the sequence.
        out:
            * is_compatible (bool)
            * comment (str) comment on why mutation is not compatible
        """
        
        # Verify if mutation position is in sequence
        if not (1 <= mutation.position <= len(self)):
            return False, f"mutation position={mutation.position} is outside of the range of sequence (L={len(self)})"
        
        # Verify if wild-type amino acid corresponds to sequence
        if mutation.wt_aa.one != self.sequence[mutation.position-1]:
            return False, f"mutation wt-aa ({mutation.wt_aa.one}) does not match aa in sequence ({self.sequence[mutation.position-1]})"
        
        # All ok
        return True, "mutation match to sequence"
    
    def list_all_single_mutations(self) -> List[str]:
        """List all possible single-site mutations on sequence (including trivial mutations like 'M13M').
        out:
            mutations (List[str]) like ['M1A', 'M1C', 'M1D', ..., 'A2A', 'A2C', ...]
        """
        mutations: List[str] = []
        for i, wt_str in enumerate(self.sequence):
            if wt_str == self.GAP_CHAR:
                continue
            for mt in self.ALL_AAS:
                mt_str = mt.one
                mutation = f"{wt_str}{str(i+1)}{mt_str}"
                mutations.append(mutation)
        return mutations
    
    # IO Methods ---------------------------------------------------------------
    def write(self, fasta_path: str) -> "Sequence":
        """Save sequence in a FASTA file."""

        # Guardians
        fasta_path = os.path.abspath(fasta_path)
        assert fasta_path.endswith(".fasta"), f"ERROR in Sequence('{self.name}').write(): fasta_path='{fasta_path}' should end with '.fasta'."

        # Create output directory if it does not exist
        os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
        
        # Save FASTA and return self
        with open(fasta_path, "w") as fs:
            fs.write(self.to_fasta_string())
        return self
