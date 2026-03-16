
# Imports ----------------------------------------------------------------------
from typing import Literal, Union
import numpy as np
from numpy.typing import NDArray
from structuredca.utils import Logger
from structuredca.dca_model.data_structures import SparseMatrix, SparseJ

# Main -------------------------------------------------------------------------
class Gauge:
    """
    Class to memorize and fix Gauge state for a Potts model.
    """

    # Constants ----------------------------------------------------------------
    LATTICE_GAS = "lattice-gas"
    ZERO_SUM = "zero-sum"
    POSSIBLE_STATES = [LATTICE_GAS, ZERO_SUM]
    
    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            logger: Union[bool, Logger]=False,
        ):

        # Init state as None (Gauge not set)
        self._state: Union[None, Literal["lattice-gas", "zero-sum"]] = None

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)

    # Base properties ----------------------------------------------------------
    @property
    def state(self) -> Union[None, Literal["lattice-gas", "zero-sum"]]:
        return self._state
    
    @property
    def is_set(self) -> bool:
        return self.state is not None
    
    def __str__(self) -> str:
        return f"Gauge['{str(self.state)}']"
    
    # Set Gauge ----------------------------------------------------------------    
    def fix_lattice_gas(
            self,
            h: NDArray[np.float32],
            J: Union[NDArray[np.float32], SparseJ],
            contacts: NDArray[np.bool_]
        ) -> "Gauge":
        """Fix Gauge to 'lattice-gas Gauge': q (gap, '-') becomes the neutral 'reference' state with zero h and J values
            hi(-) = Jij(a, -) = Jij(-, a) = 0
               for fixed state q
               for all i, j, a
        """

        # Case: Gauge already fixed
        if self.state == self.LATTICE_GAS:
            self.logger.log(f" * Gauge was already fixed to: {self}")
            return self
        
        # Choose appropriate set and get functions
        # -> to work on both J Sparse and Full implementations (and without repreated 'if' statements)
        if isinstance(J, SparseJ):
            get_Jij = SparseJ.get_Jij
            set_Jij_symmetrically = SparseJ.set_symmetrically_sparse
        else:
            get_Jij = SparseJ.get_Jij
            set_Jij_symmetrically = SparseJ.set_symmetrically_full
        # -> to work on both J Sparse and Full implementations (and without repreated 'if' statements)

        # Compute Gauge Transformation
        L, q = h.shape
        K = SparseMatrix(L, q)                # Gauge term Jij(-, a)
        Ki_sum = np.zeros((L, q), np.float32) # Aggregated sum of K[:, j, a] (for optimizing update of h)
        C = np.zeros((L, L), np.float32)      # Gauge term Jij(-, -)
        g = np.zeros((L), np.float32)         # Gauge term hi(-)
        for i in range(L):
            g[i] = h[i][-1]
            for j in range(L):
                if not contacts[i, j]:
                    continue
                Jij = get_Jij(J, i, j)
                Kij = Jij[-1]
                Ki_sum[j] += Kij
                K[i, j] = Kij
                C[i, j] = Jij[-1, -1]

        # Apply Gauge Transformation
        for i in range(L):
            
            # Fix h
            h[i] += Ki_sum[i, :] - np.sum(C[i, :]) - g[i]

            # Fix J
            for j in range(i + 1, L):
                if not contacts[i, j]:
                    continue
                Jij = get_Jij(J, i, j)
                K_ij_ab = K[j ,i].reshape((q, 1)) + K[i, j]
                set_Jij_symmetrically(J, i, j, Jij - K_ij_ab + C[i, j])

        # Set and return
        self._state = self.LATTICE_GAS
        self.logger.log(f" * Gauge fixed to: {self}")
        return self
    
    def fix_zero_sum(
            self,
            h: NDArray[np.float32],
            J: Union[NDArray[np.float32], SparseJ],
            contacts: NDArray[np.bool_],
            exclude_gaps: bool
        ) -> "Gauge":
        """Fix Gauge to 'zero-sum Gauge' (or 'Ising Gauge'): average of all field and couplings is zero
            Sum_a hi(a) = Sum_a Jij(a, b) = Sum_a Jij(b, a) = 0
                for all i, j, b
        """

        # Case: Gauge already fixed
        if self.state == self.ZERO_SUM:
            self.logger.log(f" * Gauge was already fixed to: {self}")
            return self

        # Choose appropriate set and get functions
        # -> to work on both J Sparse and Full implementations (and without repreated 'if' statements)
        # -> to work on both 'exclude_gaps' or 'keep_gaps' cases (and without repreated 'if' statements)
        if isinstance(J, SparseJ):
            if not exclude_gaps:
                get_Jij = SparseJ.get_Jij
                set_Jij_symmetrically = SparseJ.set_symmetrically_sparse
            else:
                get_Jij = SparseJ.get_Jij_except_last_index
                set_Jij_symmetrically = SparseJ.set_symmetrically_except_last_index_sparse
        else:
            if not exclude_gaps:
                get_Jij = SparseJ.get_Jij
                set_Jij_symmetrically = SparseJ.set_symmetrically_full
            else:
                get_Jij = SparseJ.get_Jij_except_last_index
                set_Jij_symmetrically = SparseJ.set_symmetrically_except_last_index_full
        
        # Set h as h without last q index if exclude_gaps
        h = h[:,:-1] if exclude_gaps else h

        # Compute Gauge Transformation
        L, q = h.shape
        K = SparseMatrix(L, q)                # Gauge term Jij(-, a)
        Ki_sum = np.zeros((L, q), np.float32) # Aggregated sum of K[:, j, a] (for optimizing update of h)
        C = np.zeros((L, L), np.float32)      # Gauge term Jij(-, -)
        g = np.zeros((L), np.float32)         # Gauge term hi(-)
        for i in range(L):
            g[i] = np.mean(h[i])
            for j in range(L):
                if not contacts[i, j]:
                    continue
                Jij = get_Jij(J, i, j)
                Kij = np.mean(Jij, axis=0)
                Ki_sum[j] += Kij
                K[i, j] = Kij
                C[i, j] = np.mean(Jij)

        # Apply Gauge Transformation
        for i in range(L):
            
            # Fix h
            h[i] += Ki_sum[i, :] - np.sum(C[i, :]) - g[i]

            # Fix J
            for j in range(i + 1, L):
                if not contacts[i, j]:
                    continue
                Jij = get_Jij(J, i, j)
                K_ij_ab = K[j ,i].reshape((q, 1)) + K[i, j]
                set_Jij_symmetrically(J, i, j, Jij - K_ij_ab + C[i, j])

        # Set and return
        self._state = self.ZERO_SUM
        self.logger.log(f" * Gauge fixed to: {self}")
        return self
