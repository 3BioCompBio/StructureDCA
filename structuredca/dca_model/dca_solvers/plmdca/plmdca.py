
# Imports ----------------------------------------------------------------------
import os.path
import glob 
import ctypes
import tempfile
from typing import Tuple, List, Union
import numpy as np
from numpy.typing import NDArray
from structuredca.dca_model.dca_solvers import DCASolver
from structuredca.dca_model.data_structures import SparseJ


# PlmDCA ----------------------------------------------------------------------
class PlmDCA(DCASolver):
    """
    DCA solver PlmDCA.
    Python wrapper for Pseudo-Likelihood Maximization Direct Coupling Analysis.
        -> Solve the DCA model by approximting likelihood by pseudo-likelihood
        -> Uses backend implemented in C++ 
    """

    # Base properties ----------------------------------------------------------
    N_AAS = 20
    N_STATES = N_AAS + 1

    def __str__(self) -> str:
        """Solver name"""
        return "DCASolver[PlmDCA]()"
    
    @classmethod
    def class_name(self) -> str:
        """Solver Class name"""
        return "DCASolver[PlmDCA]"
    
    # Main ---------------------------------------------------------------------
    def compute_coefficients(self) -> Tuple[NDArray[np.float32], Union[NDArray[np.float32], SparseJ]]:
        """Compute and return DCA model coefficients h and J."""

        # Init
        L = self.msa_length
        A = self.N_STATES

        # Find C++ plmdca compiled executable file
        plmdca_path_start = os.path.join(os.path.dirname(__file__), 'lib_plmdcaBackend*')
        plmdca_so_paths  = glob.glob(plmdca_path_start)
        try:
            PLMDCA_LIB_PATH = plmdca_so_paths[0]
        except IndexError:
            error_log = "ERROR in StructureDCA::DCASolver.PlmDCA(): C++ compiled PlmDCA '.so' library not found."
            error_log += f"   * Unable to find C++ complied PlmDCA '.so' library path at '{plmdca_path_start}'."
            error_log += f"   * Please install the StructureDCA pip package or compile the C++ code."
            raise ValueError(error_log)
        
        # Flatten contacts to a vector
        self.contacts_flat: List[bool] = []
        for i in range(L - 1):
            contacts_line = self.contacts[i]
            for j in range(i + 1, L):
                self.contacts_flat.append(bool(contacts_line[j])) # convert to normal boolean to avoid bugs with C++ convertion

        # Define coefficients sizes
        self.n_couplings = int(sum(self.contacts_flat))
        self.n_h = int(L * A)                       # [L, A]: Number of h coeff (all states at all position)
        self.n_J_max = int((L * (L-1) / 2) * A**2)  # [L, L, A, A]: Number of maximal theoretical values in J (all pairs of states at all pairs of positions by diagonal is zero and summetric)
        self.n_J = int(self.n_couplings * A**2)     # sparse([L, L, A, A]): Only coupling positions are non-zero, so matrix is space
        self.n_hJ = self.n_h + self.n_J             # h and J

        # Define plmdcaBackend Interface
        self.__plmdca = ctypes.CDLL(PLMDCA_LIB_PATH)
        self.__plmdcaBackend = self.__plmdca.plmdcaBackend
        self.__plmdcaBackend.argtypes = (
            ctypes.c_ushort,               # N_STATES
            ctypes.c_char_p,               # msa_path
            ctypes.c_int,                  # msa_length
            ctypes.POINTER(ctypes.c_bool), # contacts_flat
            ctypes.c_float,                # lambda_h
            ctypes.c_float,                # lambda_J
            ctypes.c_float,                # lambda_asymptotic
            ctypes.c_bool,                 # exclude_gaps
            ctypes.c_float,                # theta_regularization
            ctypes.c_bool,                 # count_target_sequence
            ctypes.c_bool,                 # use_weights
            ctypes.c_float,                # weights_seqid
            ctypes.POINTER(ctypes.c_float),# pos_weights
            ctypes.c_int,                  # max_iterations
            ctypes.c_int,                  # num_threads
            ctypes.c_char_p,               # weights_cache_path
            ctypes.c_bool,                 # verbose
            ctypes.c_bool,                 # log_gd_steps
            ctypes.c_char_p,               # neff_tmp_path
        )
        self.__plmdcaBackend.restype = ctypes.POINTER(ctypes.c_float * self.n_hJ)
        #extern "C" void __freeFieldsAndCouplings(float*& fields_and_couplings)
        self.__freeFieldsAndCouplings = self.__plmdca.freeFieldsAndCouplings
        #self.__freeFieldsAndCouplings.argtypes = (ctypes.POINTER(ctypes.c_float),) 
        self.__freeFieldsAndCouplings.restype = None

        # Log
        self.logger.step("Run DCASolver::plmDCA (C++ backend): to solve DCA coefficients h and J.")

        # Init tmp path for Neff cache file
        neff_tmp_file = tempfile.NamedTemporaryFile()
        self.neff_tmp_path = str(neff_tmp_file.name)

        # Compute coefficients h and J with PlmDCA
        hJ = self._get_hJ_from_backend()

        # Init h (fields)
        h = np.zeros([L, A], dtype=np.float32)
        index = 0
        for i in range(L):
            for aa_i in range(A):
                h[i, aa_i] = hJ[index]
                index += 1

        # Init J (couplings)
        if self.use_sparse_J:
            J = SparseJ(L, A, logger=self.logger)
            set_symmetrically = SparseJ.set_symmetrically_sparse
        else:
            J = np.zeros((L, L, A, A), dtype=np.float32)
            set_symmetrically = SparseJ.set_symmetrically_full
        for i in range(L - 1):
            for j in range(i + 1, L):
                if not self.contacts[i, j]: # Only positions that are in couplings are given (so skip others and keep at zero)
                    continue
                Jij = np.zeros((A, A), dtype=np.float32)
                for aa_i in range(A):
                    for aa_j in range(A):
                        Jij[aa_i, aa_j] = hJ[index]
                        index += 1
                set_symmetrically(J, i, j, Jij)

        # Free RAM (is this step useless ?)
        del self.__plmdca
        del self.__plmdcaBackend
        del self.__freeFieldsAndCouplings
        del hJ

        # Read Neff
        with open(self.neff_tmp_path, "r") as fs:
            neff_lines = fs.readlines()
        neff_tmp_file.close()
        try:
            self.Neff = float(neff_lines[0])
        except:
            self.logger.warning("StructureDCA::plmDCA: failed to parse Neff from C++ code.")

        # Return
        return h, J
    
    # Dependencies -------------------------------------------------------------
    def _get_hJ_from_backend(self) -> NDArray[np.float32]:
        """Compute fields and couplings (DCA coefficients h and J) using the C++ backend of plmDCA
            Returns them as a single long vector.
        """

        # Convert to ctypes array (magic here)

        # Set contacts for C++ backend
        argtype = ctypes.POINTER(ctypes.c_bool)
        contacts_flat_ctypes = np.ascontiguousarray(self.contacts_flat).astype(np.bool_).ctypes.data_as(argtype) # WTF

        # Set pos_weights for C++ backend
        argtype = ctypes.POINTER(ctypes.c_float)
        pos_weights_ctypes = np.ascontiguousarray(self.pos_weights).astype(np.float32).ctypes.data_as(argtype) # WTF

        # Set weights_cache_path as string and pass "" (empty string) to C++ if weights_cache_path is None
        weights_cache_path_str = self.weights_cache_path if isinstance(self.weights_cache_path, str) else ""

        # Init PlmDCA link with C++ backend and compute coefficients h and J
        hJ_ptr = self.__plmdcaBackend(
            self.N_STATES,
            self.msa_path.encode('utf-8'),
            self.msa_length,
            contacts_flat_ctypes,
            self.lambda_h,
            self.lambda_J,
            self.lambda_asymptotic,
            self.exclude_gaps,
            self.theta_regularization,
            self.count_target_sequence,
            self.use_weights,
            self.weights_seqid,
            pos_weights_ctypes,
            self.max_iterations,
            self.num_threads,
            weights_cache_path_str.encode('utf-8'),
            self.logger.verbose,
            self.log_gd_steps,
            self.neff_tmp_path.encode('utf-8'),
        )
        
        # Copy to a numpy array object
        hJ_flat = np.zeros((self.n_hJ,), dtype=np.float32)
        counter = 0
        for i, h_or_J in enumerate(hJ_ptr.contents):
            hJ_flat[i] = h_or_J
            counter += 1

        # Free fields and couplings data from PlmDCABackend
        hJ_ptr_casted = ctypes.cast(hJ_ptr, ctypes.POINTER(ctypes.c_void_p))
        self.__freeFieldsAndCouplings(hJ_ptr_casted)

        # Verify coherence
        if hJ_flat.size != counter:
            error_log = "ERROR in StructureDCA::DCASolver.PlmDCA(): h and J coefficents size mismatch from the plmDCA C++ backend:"
            error_log += f" * Data size expected from plmDCA backend: {self.n_hJ}"
            error_log += f" * Data size obtained from plmDCA backend: {hJ_flat.size}"
            raise ValueError(error_log)

        return hJ_flat
