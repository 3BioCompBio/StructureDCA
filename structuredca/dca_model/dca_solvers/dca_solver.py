
# Imports ----------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
from structuredca.utils import Logger
from structuredca.dca_model.data_structures import SparseJ

# Abstract DCASolver class -----------------------------------------------------
class DCASolver(ABC):
    """Abstract container class for DCASolver (only the part that computes h and J coefficients).
    In order to use its own solver into StructureDCA, please implement a concrete class inherited from DCASolver.
    It is only required to implement its method Solver.compute_coefficients(msa_path).
    """

    def __init__(
            self,
            msa_path: str,
            contacts: NDArray[np.bool_],
            pos_weights: NDArray[np.float32],
            lambda_h: float,
            lambda_J: float,
            lambda_asymptotic: float,
            exclude_gaps: bool,
            weights_seqid: Union[None, float],
            theta_regularization: float,
            count_target_sequence: bool,
            num_threads: int,
            max_iterations: int,
            use_sparse_J: bool,
            weights_cache_path: Union[None, str],
            logger: Union[bool, Logger],
            log_gd_steps: bool,
            **kwargs,
        ):

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)
        self.log_gd_steps = log_gd_steps

        # Init arguments
        self.msa_path = msa_path
        self.contacts = contacts
        self.pos_weights = pos_weights
        self.lambda_h = lambda_h
        self.lambda_J = lambda_J
        self.lambda_asymptotic = lambda_asymptotic
        self.exclude_gaps = exclude_gaps
        self.theta_regularization = theta_regularization
        self.count_target_sequence = count_target_sequence
        self.num_threads = num_threads
        self.max_iterations = max_iterations
        self.use_sparse_J = use_sparse_J
        self.weights_cache_path = weights_cache_path
        self.kwargs = kwargs

        # Set seqid weights parameter
        self.use_weights: bool
        self.weights_seqid: float
        if weights_seqid is None or float(weights_seqid) == 1.0: # Case: weights are disabled
            self.use_weights = False
            self.weights_seqid = 1.0
        else: # Case: do compute weights with a non-degenerated weights_seqid parameter
            self.use_weights = True
            self.weights_seqid = weights_seqid

        # Properties
        self.Neff = 0.0

    @abstractmethod
    def compute_coefficients(self) -> Tuple[NDArray[np.float32], Union[NDArray[np.float32], SparseJ]]:
        """Compute the DCA coefficients h (field) and J (couplings).
            * Considering A=21 and L the length of the MSA:
            * h of shape: [L, A]
            * J of shape: [L, L, A, A]
        """
        pass

    def __str__(self) -> str:
        """Solver name"""
        return "DCASolver()"
    
    @property
    def msa_length(self) -> int:
        """Sequence length of the MSA taget sequence."""
        return len(self.pos_weights)
    
    @classmethod
    def class_name(self) -> str:
        """Solver Class name"""
        return "DCASolver"
    
    def is_standard_dca(self) -> bool:
        """Tells if the DCA model is standard DCA (which mean that it is not StructureDCA and all contacts are considered)"""

        # Init
        L = self.contacts.shape[0]

        # Verify that all contacts are True
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                if not self.contacts[i, j]:
                    return False
        
        # Verify that all positions weights are 1
        one = np.float32(1.0)
        for i in range(L):
            if self.pos_weights != one:
                return False
        
        # All tests are passed
        return True
