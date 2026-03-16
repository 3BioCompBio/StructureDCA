
# Imports ----------------------------------------------------------------------
from typing import Union, List, Tuple
import numpy as np
from numpy.typing import NDArray
from structuredca.utils import Logger, memory_str


# SparseJ ----------------------------------------------------------------------
class SparseJ:
    """
    SparseJ(): memory efficient class to contain sparse DCA couplings (J) coefficients.
        * mimic a (L, L, q, q)-matrix but only Jij = J[i, j, :, :] that are non-zero are stored and only for i < j (others half is found by transposition)
        * J[i, i, :, :] = 0
        * J[i, j, a, b] = J[j, i, b, a] (Jij = Jji^T)

    usage:
        J = SparseJ(100, 21)
        J.set_symmetrically(13, 12, J13_12) # set a (21, 21)-matrix for position (13, 12) and its transpose to position (12, 13)
        J_12_13 = J[12, 13] # get the (21, 21)-matrix at position (12, 13)
    """

    # Constructor --------------------------------------------------------------
    def __init__(self, L: int, q: int, logger: Union[bool, Logger]=False):
        """Init (as all zeros) sparse J matrix for DCA on sequences of length L."""

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)

        # Init base properties
        self.L = L
        self.q = q
        self.ij_shape = (q, q)
        self.shape = (L, L, q, q)

        # Init sparse J data
        self._index = np.zeros((self.L, self.L), dtype=int)
        self._Jij_matrices: List[NDArray[np.float32]] = [np.zeros(self.ij_shape, dtype=np.float32)]


    # Base methods -------------------------------------------------------------
    def __str__(self) -> str:
        return f"SparseJ({self.shape}, fill={len(self._Jij_matrices)-1})"
    
    @property
    def dtype(self) -> np.dtype:
        return self[0, 0].dtype

    @staticmethod
    def _is_index_pair(ij_tuple) -> bool:
        """Return if input is of type Tuple[int, int]."""
        return isinstance(ij_tuple, tuple) and len(ij_tuple) == 2 and isinstance(ij_tuple[0], int) and isinstance(ij_tuple[1], int)
    
    def get(self, i: int, j: int) -> NDArray[np.float32]:
        """Get Jij (q, q)-NDArray."""
        Jij: NDArray[np.float32] = self._Jij_matrices[self._index[i, j]]
        if i < j:
            return Jij
        else:
            return Jij.T
        
    def __getitem__(self, ij_tuple: Tuple[int, int]) -> NDArray[np.float32]:
        """Get Jij (q, q)-NDArray."""
        assert self._is_index_pair(ij_tuple), f"ERROR in StructureDCA.SparseJ()[ij_tuple={ij_tuple}]: input ij_tuple shoud be of type Tuple[int, int]."
        i, j = ij_tuple
        return self.get(i, j)
    
    def set_symmetrically(self, i: int, j: int, Jij: NDArray[np.float32]) -> None:
        """Set Jij (q, q)-NDArray at position (i, j) and its transpose Jij^T as position (j, i)."""

        # Guardians
        assert i != j, f"ERROR in StructureDCA.SparseJ().set_symmetrically({i}, {j}, Jij): Jij matrix should remain zero when i = j."
        assert Jij.shape == self.ij_shape, f"ERROR in StructureDCA.SparseJ().set_symmetrically({i}, {j}, Jij): shape of Jij={Jij.shape} should be {self.ij_shape}."
        assert Jij.dtype == np.float32, f"ERROR in StructureDCA.SparseJ().set_symmetrically({i}, {j}, Jij): dtype of Jij={Jij.dtype} should be {np.float32}."

        # Transpose Jij matrix if (i, j) in lower half of matrix
        Jij_ = Jij if i < j else Jij.T

        # Case 1: Jij matrix was already set
        ij_index = self._index[i, j]
        if ij_index != 0:
            self._Jij_matrices[ij_index] = Jij_
            
        # Case 2: Jij matrix was not yet set
        else:
            ij_index = len(self._Jij_matrices)
            self._index[i, j] = ij_index
            self._index[j, i] = ij_index
            self._Jij_matrices.append(Jij_)

    # WARNING: As this method assigns coefficient in the J-symmetrical way, it can be missleading and thus dangerous to leave this exposed.
    #def __setitem__(self, ij_tuple: Tuple[int, int], Jij: NDArray[np.float32]) -> None:
    #    """Set Jij (q, q)-NDArray at position (i, j) and its transpose Jij^T as position (j, i)."""
    #    assert self._is_index_pair(ij_tuple), f"ERROR in StructureDCA.SparseJ()[ij_tuple={ij_tuple}]: input ij_tuple shoud be of type Tuple[int, int]."
    #    i, j = ij_tuple
    #    return self.set_symmetrically(i, j, Jij)

    def is_set(self, i: int, j: int) -> bool:
        """Returns if sub-matrix Jij is set at position (i, j)."""
        return self._index[i, j] != 0
    

    # Very specific getter and setters -----------------------------------------
    # (those seems absurd, however they are here to perform some very specific operations in an optimized way)
    def set_symmetrically_except_last_index(self, i: int, j: int, Jij: NDArray[np.float32]) -> None:
        """
        Set Jij (q-1, q-1)-matrix at position (i, j) and its transpose Jij^T as position (j, i).
            -> update J[i, j][0:q-1, 0:q-1]
            -> leave unchanged the coefficients J[i, j][:,-1] and J[i, j][-1, :]
            -> this operation if usefull to set zero-sum Gauge, while also ignoring last aa index q for gap '-'
        """

        # Guardians
        assert i != j, f"ERROR in StructureDCA.SparseJ().set_symmetrically_except_last_index({i}, {j}, Jij): Jij matrix should remain zero when i = j."
        assert Jij.shape == (self.q-1, self.q-1), f"ERROR in StructureDCA.SparseJ().set_symmetrically_except_last_index({i}, {j}, Jij): shape of Jij={Jij.shape} should be {(self.q-1, self.q-1)}."
        assert Jij.dtype == np.float32, f"ERROR in StructureDCA.SparseJ().set_symmetrically_except_last_index({i}, {j}, Jij): dtype of Jij={Jij.dtype} should be {np.float32}."

        # Transpose Jij matrix if (i, j) in lower half of matrix
        Jij_ = Jij if i < j else Jij.T

        # Case 1: Jij matrix was already set
        ij_index = self._index[i, j]
        if ij_index != 0:
            self._Jij_matrices[ij_index][:-1, :-1] = Jij_
            
        # Case 2: Jij matrix was not yet set: forbidden for set_symmetrically_except_last_index
        else:
            Jij_full = np.zeros(self.ij_shape, dtype=np.float32)
            Jij_full[:-1, :-1] = Jij_
            ij_index = len(self._Jij_matrices)
            self._index[i, j] = ij_index
            self._index[j, i] = ij_index
            self._Jij_matrices.append(Jij_full)
        
    def get_delta_slice(self, i: int, wt_aa: int, mt_aa: int, aa_sequence: List[int]) -> NDArray[np.float32]:
        """
        Get (L,)-NDArray: (J[i, j, wt_aa, aa_j] - J[i, j, mt_aa, aa_j])_{j=1, ..., L}.
            -> where aa_j is aa_sequence[j] (aa in the reference sequence) and wt_aa and mt_aa are the amino acids to compare
        """
        slice_array = np.zeros(self.L, dtype=np.float32)
        index = self._index
        Jij_matrices = self._Jij_matrices
        for j, aa_j in enumerate(aa_sequence):
            ij_index = index[i, j]
            if ij_index == 0:
                continue
            Jij = Jij_matrices[ij_index]
            if i >= j:
                Jij = Jij.T
            slice_array[j] = Jij[wt_aa, aa_j] - Jij[mt_aa, aa_j]
        return slice_array
    
    
    # Memory metrics -----------------------------------------------------------
    def size_sparse_bytes(self) -> int:
        """Memory (RAM) size of the current SparseJ object (in bytes)."""
        index_size = self.L * self.L * 8 # L*L is number of elements in index, 8 is number of bytes in int64
        Jij_size = self.q * self.q * 4   # q*q is number of elements in each Jij, 4 is number of bytes in a float32
        return index_size + len(self._Jij_matrices) * Jij_size

    def size_full_bytes(self) -> int:
        """Memory (RAM) size of J if it would be a full-matrix object (in bytes)."""
        Jij_size = self.q * self.q * 4   # q*q is number of elements in each Jij, 4 is number of bytes in a float32
        return self.L * self.L * Jij_size


    # IO methods ---------------------------------------------------------------
    def to_numpy(self) -> NDArray[np.float32]:
        """
        Return the full J (L, L, q, q)-matrix as numpy NDArray.
            WARNING: can be very memory expensive for large L (about 15 GB for L=3000).
        """

        # High RAM usage warning
        ONE_GB = 1024**3
        size_full_bytes = self.size_full_bytes()
        if size_full_bytes > ONE_GB:
            size_full_str = memory_str(size_full_bytes)
            self.logger.warning(f"high RAM usage: SparseJ().to_numpy() will create a {self.shape}-matrix of {size_full_str} RAM.")

        # Create full J matrix
        J = np.zeros(self.shape, dtype=np.float32)
        for i in range(self.L):
            for j in range(self.L):
                if self.is_set(i, j):
                    J[i, j] = self.get(i, j)
        return J


    # Utility "polymorphic" functions --------------------------------------------------------
    # Functoins to get and set Jij on both Sparse and Full J

    @staticmethod
    def set_symmetrically_sparse(J: "SparseJ", i: int, j: int, Jij: NDArray[np.float32]) -> None:
        """Set Jij (q, q)-NDArray at position (i, j) and its transpose Jij^T as position (j, i) for Sparse J."""
        J.set_symmetrically(i, j, Jij)

    @staticmethod
    def set_symmetrically_full(J: NDArray[np.float32], i: int, j: int, Jij: NDArray[np.float32]) -> None:
        """Set Jij (q, q)-NDArray at position (i, j) and its transpose Jij^T as position (j, i) for Full NDArray J."""
        J[i, j] = Jij
        J[j, i] = Jij.T

    @staticmethod
    def set_symmetrically_except_last_index_sparse(J: "SparseJ", i: int, j: int, Jij_cropped: NDArray[np.float32]) -> None:
        """Set Jij (q-1, q-1)-NDArray at position (i, j) and its transpose Jij^T as position (j, i) for Sparse J.
            -> Leave last index coefficients J[i, j, :, q] and J[i, j, q, :] unchanged (they correspond to gap '-')
        """
        J.set_symmetrically_except_last_index(i, j, Jij_cropped)

    @staticmethod
    def set_symmetrically_except_last_index_full(J: NDArray[np.float32], i: int, j: int, Jij_cropped: NDArray[np.float32]) -> None:
        """Set Jij (q-1, q-1)-NDArray at position (i, j) and its transpose Jij^T as position (j, i) for Sparse J.
         -> Leave last index coefficients J[i, j, :, q] and J[i, j, q, :] unchanged (they correspond to gap '-')
        """
        J[i, j][:-1, :-1] = Jij_cropped
        J[j, i][:-1, :-1] = Jij_cropped.T

    @staticmethod
    def get_Jij(J: Union[NDArray[np.float32], "SparseJ"], i: int, j: int) -> NDArray[np.float32]:
        """Return Jij (q, q)-NDArray at position (i, j) of J (for Sparse or Full J)."""
        return J[i, j]
    
    @staticmethod
    def get_Jij_except_last_index(J: Union[NDArray[np.float32], "SparseJ"], i: int, j: int) -> NDArray[np.float32]:
        """Return Jij (q-1, q-1)-NDArray at position (i, j) of J (for Sparse or Full J).
            -> ignores last index Jij[:, q] and Jij[q, :] (they correspond to gap '-')
        """
        return J[i, j][:-1 ,:-1]
