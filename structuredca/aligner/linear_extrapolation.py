
# Imports ----------------------------------------------------------------------
from typing import List
import numpy as np
from numpy.typing import NDArray

# Main -------------------------------------------------------------------------
class LinearExtrapolation:
    """
    Class to extrapolate distance matrices distance_increment between two consecutive nodes.

    args:
        distance_increment:     float    increment distance between two nodes: d[i, i+2] - d[i, i+1]
        distance_adj: float    distance_increment for two neighbour nodes: d[i, i+1]

    usage:
        lin_ext = LinearExtrapolation(distance_increment=3.2, distance_adj=0.135)

        squared_dist_matrix = lin_ext.extrapolate_diagonal_matrix(10) # shape = (10, 10)

        prolongation_dist_matrix = lin_ext.extrapolate_marginal_matrix([5.1, 5.3, 5.0], 10) # shape = (10, 3)
    """

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            distance_increment: float,
            distance_adj: float,
            name: str="linear_extrapolator"
        ):

        # Guardians
        assert distance_increment > 0.0, f"ERROR in LinearExtrapolation(): distance_increment={distance_increment} should be stricktly positive."
        assert distance_adj > 0.0, f"ERROR in LinearExtrapolation(): distance_adj={distance_adj} should be stricktly positive."

        # Init properties
        self.name = name
        self.distance_increment = np.float32(distance_increment)
        self.distance_adj = np.float32(distance_adj)

    # Base methods -------------------------------------------------------------
    def __str__(self) -> str:
        return f"LinearExtrapolation('{self.name}', d_inc={self.distance_increment:.2f}, d_nxt={self.distance_adj:.2f})"
    
    # Extrapoation -------------------------------------------------------------    
    def extrapolate_diagonal_matrix(self, n: int) -> NDArray[np.float32]:
        """
        Return extrapolated diagonal distance matrix (shape=(n, n)).
        """
        assert n > 0, f"ERROR in {self}.extrapolate_diagonal_matrix(): n='{n}' should be stricktly positive."
        extrapolation_array = [self.distance_adj + i*self.distance_increment for i in range(n)]
        matrix = np.zeros((n, n), dtype=np.float32)
        for i1 in range(n):
            for i2 in range(i1):
                delta_i = i1 - i2
                d = extrapolation_array[delta_i - 1]
                matrix[i1, i2] = d
                matrix[i2, i1] = d
        return matrix
    
    def extrapolate_marginal_matrix(self, distance_array: List[float], n: int, reverse_lines: bool=False) -> NDArray[np.float32]:
        """
        Return extrapolated marginal distance matrix by extrapolating n steps from a distance_array (shape=(n, len(distance_array))).
        """

        # Guardians
        assert n > 0, f"ERROR in {self}.extrapolate_marginal_matrix(): n='{n}' should be stricktly positive."
        
        # Init
        ZERO = np.float32(0.0)
        distance_array = np.array(distance_array, dtype=np.float32)
        
        # First increment (might use distance_adj or distance_increment)
        first_increment_arr = np.array([self.distance_adj if d == ZERO else self.distance_increment for d in distance_array], dtype=np.float32)
        matrix = [distance_array + first_increment_arr]
        
        # Following increments (use distance_increment)
        for i in range(n-1):
            increment = np.float32((i+2)*self.distance_increment)
            matrix.append(distance_array + increment)

        # Reverse if required
        if reverse_lines:
            matrix = matrix[::-1]

        # Format and return
        return np.array(matrix, dtype=np.float32)
