
# Imports ----------------------------------------------------------------------
import os.path
import gzip
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure as BPStructure

# Main -------------------------------------------------------------------------
class StructureReader:
    """3D structure reader that returns a BioPython Structure object."""

    # Constants ----------------------------------------------------------------
    ACCEPTED_EXTENTIONS = [
        "pdb", "ent", "cif",
        "pdb.gz", "ent.gz", "cif.gz",
    ]
    READ_FILE_ENCODING = "ISO-8859-1" # prevents read bugs in some cases, wtf

    # Methods ------------------------------------------------------------------
    @classmethod
    def read_biopython_structure(cls, pdb_path: str) -> BPStructure:
        """Return a BioPython Structure object from 'pdb_path' path to a 3D structure file.
            - handle '.pdb', '.ent' and '.cif' formats
            - handle '.gz' compressed files
        """

        # Guardians
        pdb_path = str(pdb_path)
        if not cls.has_valid_extention(pdb_path):
            raise ValueError(
                f"ERROR in StructureReader.read_biopython_structure(): "
                f"pdb_path='{pdb_path}' should end with any of {cls.ACCEPTED_EXTENTIONS}."
            )

        # Select parser
        if pdb_path.endswith(".cif") or pdb_path.endswith(".cif.gz"):
            pdb_parser = MMCIFParser(QUIET=True)
        else:
            pdb_parser = PDBParser(QUIET=True, get_header=True)

        # Select file handler
        if pdb_path.endswith(".gz"):
            custom_open = gzip.open
        else:
            custom_open = open

        # Parse structure with biopython
        pdb_name = cls.get_pdb_name(pdb_path)
        with custom_open(pdb_path, mode="rt", encoding=cls.READ_FILE_ENCODING) as fs:
            bp_structure: BPStructure = pdb_parser.get_structure(pdb_name, fs)
        return bp_structure

    @classmethod
    def has_valid_extention(cls, pdb_path: str) -> bool:
        """Return True if 'pdb_path' has a file extention among Structure.ACCEPTED_EXTENTIONS."""
        pdb_path = str(pdb_path)
        return any([pdb_path.endswith(f".{ext}") for ext in cls.ACCEPTED_EXTENTIONS])

    @classmethod
    def get_pdb_name(clf, pdb_path: str) -> str:
        """Get 'pdb_name' from 'pdb_path' by removing file extention and directory prefix."""
        pdb_name = os.path.basename(str(pdb_path))
        for ext in clf.ACCEPTED_EXTENTIONS:
            if pdb_name.endswith(f".{ext}"):
                pdb_name = pdb_name.removesuffix(f".{ext}")
                break
        return pdb_name
