
# Imports ----------------------------------------------------------------------
import os.path
from typing import List
from structuredca.utils import is_convertable_to
from structuredca.sequence import AminoAcid

# Mutation ---------------------------------------------------------------------
class Mutation:
    """Container class for a single missence amino acid mutation (residue ID in fasta format)."""

    # Constructor --------------------------------------------------------------
    def __init__(self, wt_aa: AminoAcid, position: int, mt_aa: AminoAcid):
        self.wt_aa = wt_aa
        self.position = position
        self.mt_aa = mt_aa

    @classmethod
    def parse_single_mutation(self, mutation_str: str) -> "Mutation":
        """Parse a Mutation from a string like 'M13A'"""

        # Unpack and guardians
        assert len(mutation_str) >= 3, f"ERROR in Mutation.parse_single_mutation(): invalid mutation_str='{mutation_str}': should be of length 3 or more."
        wt_str, position_str, mt_str = mutation_str[0], mutation_str[1:-1], mutation_str[-1]
        assert AminoAcid.one_exists(wt_str), f"ERROR in Mutation.parse_single_mutation(): invalid mutation_str='{mutation_str}': wild-type amino acid is incorrect."
        assert AminoAcid.one_exists(mt_str), f"ERROR in Mutation.parse_single_mutation(): invalid mutation_str='{mutation_str}': mutant amino acid is incorrect."
        assert is_convertable_to(position_str, int) and int(position_str) > 0, f"ERROR in Mutation(): invalid mutation_str='{mutation_str}': position must be a stricktly positive integer."

        # Construct and return
        return Mutation(AminoAcid(wt_str), int(position_str), AminoAcid(mt_str))
    
    @classmethod
    def parse_mutations_list(self, mutations_str: str) -> List["Mutation"]:
        """Parse a list of Mutations from a string like 'M13A:G15K'"""

        # Convert to mutations list
        mutations_list = [Mutation.parse_single_mutation(mut) for mut in mutations_str.split(":")]

        # Positional coherence check
        unique_positions = set([mut.position for mut in mutations_list])
        if len(unique_positions) != len(mutations_list):
            raise ValueError(f"ERROR in Mutation.parse_mutations_list(): mutations_str='{mutations_str}' contains redundent positions.")
        
        # Return
        return mutations_list

    # Methods ------------------------------------------------------------------
    def __str__(self) -> str:
        """Return standard string representation of a Mutation like 'M13A' (wt_aa-residue_id-mt_aa)"""
        return f"{self.wt_aa.one}{self.position}{self.mt_aa.one}"
    
    def __int__(self) -> int:
        """Return unique integer code for each mutation."""
        return self.position*10000 + self.wt_aa.id*100 + self.mt_aa.id
    
    def is_trivial(self) -> bool:
        """Return if mutation is trivial (like 'A14A')."""
        return self.wt_aa == self.mt_aa

# Mutation utils functions -----------------------------------------------------
def read_mutations_file(mutations_file_path: str) -> List[str]:
    """Read a list of mutations from a file.
        - file should contain one mutation (single or multiple) by line
        - examples of a correct mutation: 'A54H' or 'M13A:G15K'
    """

    # Example of correct mutations_file
    correct_mutations_file_str = (
        "Example of a valid mutations file:\n"
        "'''\n"
        "A54H\n"
        "M13A:G15K\n"
        "'''"   
    )

    # Guadian
    assert os.path.isfile(mutations_file_path), f"ERROR in read_mutations_file(): mutations_file_path='{mutations_file_path}' file does not exist.\n{correct_mutations_file_str}"

    # Parse mutations
    with open(mutations_file_path, "r") as fs:
        mutations_list: List[str] = [line.strip() for line in fs.readlines()]
    assert len(mutations_list) > 0, f"ERROR in read_mutations_file(): mutations_file_path='{mutations_file_path}' file is empty.\n{correct_mutations_file_str}"

    # Validate mutations
    for i, mutation_str in enumerate(mutations_list):
        try:
            _ = Mutation.parse_mutations_list(mutation_str)
        except Exception as err:
            msg = (
                f"ERROR in read_mutations_file(mutations_file_path='{mutations_file_path}'): \n"
                f" -> in mutations {i+1} / {len(mutations_list)}: '{mutation_str}'\n"
                f" -> {err}\n"
                f"{correct_mutations_file_str}"
            )
            raise ValueError(msg)

    # Return
    return mutations_list
