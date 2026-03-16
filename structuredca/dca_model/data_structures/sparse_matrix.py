
# Imports ----------------------------------------------------------------------
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray


# SparseMatrix -----------------------------------------------------------------
class SparseMatrix:
    """
    SparseMatrix(): memory efficient class to contain a Sparse Matrix of shape (L, L, q).

    usage:
        M = SparseMatrix(100, 21) # shape = (100, 100, 21)
        M[13, 12] = M13_12    # set a (q,)-NDArray to position (13, 12)
        M13_12 = M[13, 12]    # get the (q,)-NDArray at position (13, 12)
    """

    # Constructor --------------------------------------------------------------
    def __init__(self, L: int, q: int):
        """Init (as all zeros) sparse M matrix of shape (L, L, q)."""

        # Init base properties
        self.L = L
        self.q = q
        self.ij_shape = (q,)
        self.shape = (L, L, q)

        # Init sparse M data
        self._index = np.zeros((self.L, self.L), dtype=int)
        self._Mij_matrices: List[NDArray[np.float32]] = [np.zeros(self.ij_shape, dtype=np.float32)]


    # Base methods -------------------------------------------------------------
    def __str__(self) -> str:
        return f"SparseMatrix({self.shape}, fill={len(self._Mij_matrices)-1})"
    
    @property
    def dtype(self) -> np.dtype:
        return self[0, 0].dtype

    @staticmethod
    def _is_index_pair(ij_tuple) -> bool:
        """Return if input is of type Tuple[int, int]."""
        return isinstance(ij_tuple, tuple) and len(ij_tuple) == 2 and isinstance(ij_tuple[0], int) and isinstance(ij_tuple[1], int)

    def get(self, i: int, j: int) -> NDArray[np.float32]:
        """Get Mij (q,)-NDarray object from position (i, j)."""
        return self._Mij_matrices[self._index[i, j]]
        
    def __getitem__(self, ij_tuple: Tuple[int, int]) -> NDArray[np.float32]:
        """Get Mij (q,)-NDarray object from position (i, j)."""
        assert self._is_index_pair(ij_tuple), f"ERROR in StructureDCA.SparseMatrix()[ij_tuple={ij_tuple}]: input ij_tuple shoud be of type Tuple[int, int]."
        i, j = ij_tuple
        return self.get(i, j)
    
    def set(self, i: int, j: int, Mij: NDArray[np.float32]) -> None:
        """Set Mij (q,)-NDarray object to position (i, j)."""

        # Guardians
        assert Mij.shape == self.ij_shape, f"ERROR in StructureDCA.SparseMatrix().set({i}, {j}, Mij): shape of Mij={Mij.shape} should be {self.ij_shape}."
        assert Mij.dtype == np.float32, f"ERROR in StructureDCA.SparseMatrix().set({i}, {j}, Mij): dtype of Mij={Mij.dtype} should be {np.float32}."

        # Case 1: Jij matrix was already set
        ij_index = self._index[i, j]
        if ij_index != 0:
            self._Mij_matrices[ij_index] = Mij
            
        # Case 2: Jij matrix was not yet set
        else:
            ij_index = len(self._Mij_matrices)
            self._index[i, j] = ij_index
            self._Mij_matrices.append(Mij)

    def __setitem__(self, ij_tuple: Tuple[int, int], Mij: NDArray[np.float32]) -> None:
        """Set Mij (q,)-NDarray object to position (i, j)."""
        assert self._is_index_pair(ij_tuple), f"ERROR in StructureDCA.SparseMatrix()[ij_tuple={ij_tuple}]: input ij_tuple shoud be of type Tuple[int, int]."
        i, j = ij_tuple
        return self.set(i, j, Mij)

    def is_set(self, i: int, j: int) -> bool:
        """Returns if (q,)-NDarray Mij is set for position (i, j)."""
        return self._index[i, j] != 0
