
# Imports ----------------------------------------------------------------------
import os.path
from typing import List, Tuple, Dict, Any, Literal, Union
import tempfile
import numpy as np
from numpy.typing import NDArray
from structuredca.utils import Logger, format_str
from structuredca.sequence import AminoAcid, Sequence, FastaReader, MSA, Mutation
from structuredca.dca_model.dca_solvers import DCASolver, PlmDCA
from structuredca.dca_model.gauge import Gauge
from structuredca.dca_model.data_structures import SparseJ


# DCAModel DCA Solver ----------------------------------------------------------
class DCAModel:
    """
    Class to generate and handle DCA model (h, J) and call a DCASolver.
        * Computes h and J weights from an MSA using the DCASolver.
        * Allows to write/read coefficients h and J and hyperparameters to a file using the NumPy '.npz' format.
        * Compute evolutionary energy values X for a sequence of a different in evolutionary energy dX for a mutation(s)
    """

    # Constructor --------------------------------------------------------------
    
    # Constants
    ALL_STATES: List[str] = [aa.one for aa in AminoAcid.get_all()] + [AminoAcid.GAP_ONE]
    N_STATES = len(ALL_STATES)
    DCA_SOLVERS: Dict[str, DCASolver] = {
        "plmDCA": PlmDCA,
    }

    # Init class
    def __init__(
            self,
            msa_path: str,
            name: str,
            contacts: NDArray[np.bool_],
            wh: NDArray[np.float32],
            wJ: NDArray[np.float32],
            lambda_h: float,
            lambda_J: float,
            lambda_asymptotic: float,
            exclude_gaps: bool,
            min_seqid: Union[None, float],
            weights_seqid: Union[None, float],
            contacts_gap_cutoff: Union[None, float],
            theta_regularization: float,
            count_target_sequence: bool,
            num_threads: int,
            solver: Union[Literal["plmDCA"], DCASolver]="plmDCA",
            max_iterations: int=2000,
            use_sparse_J: bool=True,
            weights_cache_path: Union[None, str]=None,
            logger: Union[bool, Logger]=True,
            log_gd_steps: bool=False,
        ):

        # Init logger
        if isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = Logger(verbose=logger)
        self.log_gd_steps = log_gd_steps

        # DCA Solver guaridans
        solvers_list = list(self.DCA_SOLVERS.keys())
        if isinstance(solver, str):
            assert solver in solvers_list, f"ERROR in DCAModel(): DCA solver='{solver}' does not exist. Existing solvers: {solvers_list}."
        else:
            assert issubclass(solver, DCASolver), f"ERROR in DCAModel(): DCA Solver='{solver}' should be a string in {solvers_list} or a class inherited from DCASolver."

        # Init solver
        if isinstance(solver, str):
            self.solver_name = solver
            self.solver_class = self.DCA_SOLVERS[solver]
        else:
            self.solver_name = solver.class_name()
            self.solver_class = solver

        # Param Guardians
        assert os.path.isfile(msa_path), f"ERROR in DCAModel(): msa_path='{msa_path}' file does not exists."
        assert lambda_h > 0.0, f"ERROR in DCAModel(): lambda_h={lambda_h} should be stricktly positive."
        assert lambda_J > 0.0, f"ERROR in DCAModel(): lambda_J={lambda_J} should be stricktly positive."
        assert lambda_asymptotic >= 0.0, f"ERROR in DCAModel(): lambda_asymptotic={lambda_asymptotic} should be positive."
        assert 0.0 < theta_regularization < 1.0, f"ERROR in DCAModel(): theta_regularization={theta_regularization} should be stricktly in [0, 1]."
        if weights_seqid is not None:
            assert 0.0 <= weights_seqid <= 1.0, f"ERROR in DCAModel(): weights_seqid={weights_seqid} shoud be in [0, 1] interval."
        if min_seqid is not None:
            assert 0.0 <= min_seqid < 1.0, f"ERROR in DCAModel(): min_seqid={min_seqid} should be in [0, 1["
        assert max_iterations >= 0, f"ERROR in DCAModel(): max_iterations={max_iterations} should be positive."
        assert num_threads > 0, f"ERROR in DCAModel(): num_threads={num_threads} should be stricktly positive."
        if weights_cache_path is not None:
            assert isinstance(weights_cache_path, str), f"ERROR in DCAModel(): weights_cache_path='{weights_cache_path}' should be a string or None."

        # Init all hyperparameters
        self.msa_path = msa_path
        self.name = name
        self.contacts = contacts
        self.lambda_h = lambda_h
        self.lambda_J = lambda_J
        self.lambda_asymptotic = lambda_asymptotic
        self.exclude_gaps = exclude_gaps
        self.min_seqid = min_seqid
        self.weights_seqid = weights_seqid
        self.contacts_gap_cutoff = contacts_gap_cutoff
        self.theta_regularization = theta_regularization
        self.count_target_sequence = count_target_sequence
        self.num_threads = num_threads
        self.max_iterations = max_iterations
        self.use_sparse_J = use_sparse_J
        self.weights_cache_path = weights_cache_path

        # Set base MSA properties
        target_sequence: Sequence = FastaReader.read_first_sequence(msa_path)
        assert not target_sequence.contains_gaps(), f"ERROR in DCAModel(): target sequence of the MSA can not contains gaps."
        assert target_sequence.is_all_amino_acids(), f"ERROR in DCAModel(): target sequence of the MSA can not contains non-standard amino acids."
        self.msa_length = len(target_sequence)
        self.target_sequence = target_sequence
        self.target_sequence_str = target_sequence.sequence
        self.target_sequence_int = self._encode_sequence_to_ids(self.target_sequence_str)
        self.length_ids = [i for i in range(self.msa_length)]
        self.msa_depth = 0
        self.msa_initial_depth = 0
        self.Neff = 0.0

        # Init h and J weights
        L = self.msa_length
        self.plm_positions_weights = np.ones(L, np.float32) # weight positions in the PLM cost function when solving DCA coefficients
        self.wh = wh
        self.wJ = wJ

        # Init DCA properties
        self.is_initialized: bool = False   # Will be set to True when h and J are solved
        self.h: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.J: SparseJ = SparseJ(L, self.N_STATES, logger=self.logger)

        # Init Gauge
        self.gauge = Gauge(logger=self.logger)

    # Init and solve DCA model: h and J coefficients
    def init_dca(self) -> "DCAModel":
        """Solve DCA coefficients h and J."""
        
        # Log
        self.logger.step(f"Solve DCA model (h and J coefficients).")

        # Check consistency of contact matrix (in case user edited it)
        if not np.array_equal(self.contacts, self.contacts.T):
            raise ValueError(f"ERROR in {self}: contacts matrix should be symmetric (C[i, j] = C[j, i]).")
        if not np.all(self.contacts.diagonal() == False):
            raise ValueError(f"ERROR in {self}: contacts matrix diagonal should be False (C[i, i] = False).")

        # Init DCA
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Pre-process MSA
            tmp_msa_path = os.path.join(tmp_dir, f"{self.name}.fasta")
            msa_processed = MSA(self.msa_path, min_seqid=self.min_seqid, logger=self.logger)
            msa_processed.write(tmp_msa_path)
            self.msa_depth = msa_processed.depth
            self.msa_initial_depth = msa_processed.initial_depth
            self.gap_ratios_array = msa_processed.gap_ratios_positions()
            
            # Set contacts of all positions with gap ratio >= threshold to False
            self._remove_contacts_by_gap_cutoff()

            # Free RAM
            del msa_processed

            # Initialize fields and couplings: h and J with a DCASolver
            dca_solver: DCASolver = self.solver_class(
                tmp_msa_path,
                self.contacts,
                self.plm_positions_weights,
                self.lambda_h,
                self.lambda_J,
                self.lambda_asymptotic,
                self.exclude_gaps,
                self.weights_seqid,
                self.theta_regularization,
                self.count_target_sequence,
                self.num_threads,
                self.max_iterations,
                self.use_sparse_J,
                self.weights_cache_path,
                self.logger,
                self.log_gd_steps,
            )
            self.h, self.J = dca_solver.compute_coefficients()
            self.is_initialized = True
            self.Neff = dca_solver.Neff

        # Fix zero-sum Gauge
        self.logger.step("Fix Gauge.")
        self.fix_zero_sum_gauge()
        return self
    
    def _remove_contacts_by_gap_cutoff(self) -> None:
        """Remove contacts at positions where gap-ratio is higher than this contacts_gap_cutoff."""

        # Guardians
        if self.contacts_gap_cutoff is not None:
            assert 0.0 <= self.contacts_gap_cutoff <= 1.0, f"ERROR in DCAModel._remove_contacts_by_gap_cutoff(): contacts_gap_cutoff={self.contacts_gap_cutoff} should be between 0 and 1."

        # Skip if contacts_gap_cutoff is trivial
        if self.contacts_gap_cutoff is None or self.contacts_gap_cutoff >= 1.0:
            return None

        # Remove contacts
        n_contacts1 = self.n_contacts()
        for i, gap_ratio in enumerate(self.gap_ratios_array):
            if gap_ratio <= self.contacts_gap_cutoff:
                continue
            self.contacts[i, :] = False
            self.contacts[:, i] = False
        n_contacts2 = self.n_contacts()

        # Log
        self.logger.log(f" * remove contacts at highly gapped positions (gap-ratio > {self.contacts_gap_cutoff:.2f}): {n_contacts1} -> {n_contacts2}")  

    # Base properties and methods ----------------------------------------------
    def __str__(self) -> str:
        return f"DCAModel('{self.name}', [l={self.msa_length}, d={self.msa_depth}], s='{self.solver_name}')"
    
    def show(self) -> "DCAModel":
        """Show hyperparameters of DCAModel"""
        status = "DCA-initialized" if self.is_initialized else "DCA-not-initialized"
        print(f"DCAModel (status: {status})")
        hyperparameters = self.export_hyperparameters()
        for param_name, param_value in hyperparameters.items():
            print(f"   * {param_name}: {format_str(param_value)}")
        return self
    
    def n_contacts(self) -> int:
        """Total number of contacts between pairs of positions (count twice (i, j) and (j, i) but not diagonal (i, i))."""
        return int(np.sum(self.contacts))
    
    def n_contacts_max(self) -> int:
        """Maximum possible number of contacts (L² - L)."""
        return self.msa_length**2 - self.msa_length
    
    def contacts_ratio(self) -> float:
        """Ratio of effective contacts among all possible contacts."""
        return self.n_contacts() / self.n_contacts_max()
    
    def avg_contacts(self) -> float:
        """Average number of contacts by position."""
        return self.n_contacts() / self.msa_length

    def fix_lattice_gas_gauge(self) -> "DCAModel":
        """Fix Gauge to 'lattice-gas Gauge': q (gap, '-') becomes the neutral 'reference' state with zero h and J values
            hi(-) = Jij(a, -) = Jij(-, a) = 0     for all i, j, a
        """
        self.gauge.fix_lattice_gas(self.h, self.J, self.contacts)
        return self

    def fix_zero_sum_gauge(self) -> "DCAModel":
        """Fix Gauge to 'zero-sum Gauge' (or 'Ising Gauge'): average of all field and couplings is zero
            Sum_a hi(a) = Sum_a Jij(a, b) = Sum_a Jij(b, a) = 0     for all i, j, b
        """
        self.gauge.fix_zero_sum(self.h, self.J, self.contacts, self.exclude_gaps)
        return self

    # DCA methods --------------------------------------------------------------
    def position_probabilities(
            self,
            position: int,
            reweight_by_rsa: bool=False,
            background_sequence: Union[None, str]=None,
            temperature: float=1.0,
        ) -> Dict[str, np.float32]:
        """Return the relative probabilities for the 20 standard Amino Acids at this position.

        Args:
            position (int):     0-based index of the position in the MSA
                               (WARNING: not like in FASTA format where it starts at 1).
            reweight_by_rsa (bool=False):     use RSA-complement as weights wh and wJ
            background_sequence (str=None):   alternative target sequence to mutate from
            temperature (float=1.0):          temperature T as in the Boltzmann distribution P ~ exp(-E/T)
        Returns:
            position_probabilities (dict[str, np.float32]): amino acid (1-letter-code) -> P(i, AA)
        """

        # Init
        if temperature <= 0.0:
            raise ValueError("Input temperature can not be negative or zero.")
        if background_sequence is None:
            wt = self.target_sequence[position]
        else:
            wt = background_sequence[position]
        fasta_position = str(position + 1) # as indexed in FASTA format starting at 1
        all_aas = [aa.one for aa in AminoAcid.get_all()]

        # Compute DeltaE for all AAs
        dE_array = np.array([
            self.eval_mutation(
                f"{wt}{fasta_position}{mt}",
                reweight_by_rsa=reweight_by_rsa,
                background_sequence=background_sequence,
            )
            for mt in all_aas
        ])

        # Compute probabilities
        proba_array = np.exp(-dE_array / temperature)
        proba_array = proba_array / proba_array.sum()
        return {aa: p for aa, p in zip(all_aas, proba_array)}

    def eval_mutation(
            self,
            mutation_str: str,
            reweight_by_rsa: bool=False,
            background_sequence: Union[None, str]=None,
        ) -> np.float32:
        """Evaluate evolutionary energy difference dE (wt -> mt) of a missence mutation.
        
        Args:
            mutation_str (str):               mutation to evaluate like 'M13G' or 'M13G:H15K'
            reweight_by_rsa (bool=False):     use RSA-complement as weights wh and wJ
            background_sequence (str=None):   alternative target sequence to mutate from
        Returns:
            dE (np.float32): delta evolutionary energy (dE = dh + Sum dJ[i])
                - dE > 0 means unfavorable / destabilizing mutation
        """
        dh, dJ = self.eval_mutation_hJ(mutation_str, reweight_by_rsa, background_sequence=background_sequence)
        return dh + dJ
    
    def eval_mutation_hJ(
            self,
            mutation_str: str,
            reweight_by_rsa: bool=False,
            background_sequence: Union[None, str]=None,
        ) -> Tuple[np.float32, np.float32]:
        """Evaluate missence mutation and return Delta(wt -> mt) for dh and dJ.
        
        arg:
            mutation_str (str)               mutation to evaluate like 'M13G' or 'M13G:H15K'
            reweight_by_rsa (bool=False)     use RSA-complement as weights wh and wJ
            background_sequence (str=None)   alternative target sequence to mutate from
        out:
            * dh: delta single-site effect of mutation
            * dJ: delta couplings effect of mutation (dJ = Sum dJ[i])
                - dh, dJ > 0 means unfavorable / destabilizing mutation
        """
        dh, dJarr = self.eval_mutation_hJarr(mutation_str, reweight_by_rsa, background_sequence=background_sequence)
        dJ = np.sum(dJarr)
        return dh, dJ
    
    def eval_mutation_hJarr(
            self,
            mutation_str: str,
            reweight_by_rsa: bool=False,
            background_sequence: Union[None, str]=None,
        ) -> Tuple[np.float32, NDArray[np.float32]]:
        """Evaluate missence mutation and return Delta(wt -> mt) for dh and dJ[i] array of length L.
        
        arg:
            mutation_str (str)               mutation to evaluate like 'M13G' or 'M13G:H15k'
            reweight_by_rsa (bool=False)     use RSA-complement as weights wh and wJ
            background_sequence (str=None)   alternative target sequence to mutate from
        out:
            * dh:    delta single-site effect of mutation
            * dJ[i]: delta coupling effect with position i of mutation
                - dh, dJ[i] > 0 means unfavorable / destabilizing mutation
        """

        # Parse mutations
        mutations_list = Mutation.parse_mutations_list(mutation_str)

        # Copy initial sequence as list of integers
        # -> then we will be able to 'mutate' it to perform multiple mutations
        # Here we will iteratively accumulate the dE for each mutation while 'mutating' the sequence to mutate
        # This is not optimal approach but much simple and generalizes well to any number of mutations
        if background_sequence is None:
            initial_sequence = [aa_i for aa_i in self.target_sequence_int]
        else:
            initial_sequence = self.encode_sequence_to_ids(background_sequence)

        # Single mutation case (for optimization)
        if len(mutations_list) == 1:
            return self._eval_single_mutation_hJarr(mutations_list[0], reweight_by_rsa, initial_sequence)

        # Compute effects for first mutation and initialize dh and dJarr
        first_mutation = mutations_list[0]
        dh, dJarr = self._eval_single_mutation_hJarr(first_mutation, reweight_by_rsa, initial_sequence)
        initial_sequence[first_mutation.position - 1] = first_mutation.mt_aa.id

        # Aggregate effects for following mutations
        for mutation in mutations_list[1:]:
            dh_mut, dJarr_mut = self._eval_single_mutation_hJarr(mutation, reweight_by_rsa, initial_sequence)
            initial_sequence[mutation.position - 1] = mutation.mt_aa.id
            dh += dh_mut
            dJarr += dJarr_mut

        # Return
        return dh, dJarr
    
    def _eval_single_mutation_hJarr(
            self,
            mutation: Mutation,
            reweight_by_rsa: bool,
            initial_sequence: List[int],
        ) -> Tuple[np.float32, NDArray[np.float32]]:
        """Evaluate single-site missence mutation and return Delta(wt -> mt) for dh and dJ[i] array of length L.
        
        arg:
            mutation (Mutation)               mutation to evaluate
            reweight_by_rsa (bool=False)      use RSA-complement as weights wh and wJ
            initial_sequence (List[int])      starting sequence on which to evaluate mutation
        out:
            * dh:    delta single-site effect of mutation
            * dJ[i]: delta coupling effect with position i of mutation
                - dh, dJ[i] > 0 means unfavorable / destabilizing mutation
        """

        # Guardian
        if not (1 <= mutation.position <= len(initial_sequence)):
            raise ValueError(f"Mutation '{mutation}' position is out of range of target sequence (L={self.msa_length})")
        mut_aa_id = mutation.wt_aa.id
        seq_aa_id = initial_sequence[mutation.position - 1]
        if mut_aa_id != seq_aa_id:
            mut_aa_one = AminoAcid.ID_2_ONE.get(mut_aa_id, AminoAcid.GAP_ONE)
            seq_aa_one = AminoAcid.ID_2_ONE.get(seq_aa_id, AminoAcid.GAP_ONE)
            raise ValueError(f"Mutation '{mutation}' wt-aa ('{mut_aa_one}') does not match aa in target sequence ('{seq_aa_one}')")

        # Skip if trivial mutation (for optimization)
        if mutation.wt_aa == mutation.mt_aa:
            return np.float32(0), np.zeros(self.msa_length, dtype=np.float32)

        # Compute dh and dJarr for mutation
        i = mutation.position - 1 # positional shift between fasta and python arrays
        wt_aa, mt_aa = mutation.wt_aa.id, mutation.mt_aa.id
        dh = self.h[i, wt_aa] - self.h[i, mt_aa] # we use (wt - mt) because dE = -Hamiltonian(wt -> mt)
        if isinstance(self.J, SparseJ):
            dJarr = self.J.get_delta_slice(i, wt_aa, mt_aa, initial_sequence)
        else:
            dJarr = (self.J[i, self.length_ids, wt_aa, initial_sequence] - self.J[i, self.length_ids, mt_aa, initial_sequence])

        # Reweights by RSA-complement if required
        if reweight_by_rsa:
            dh *= self.wh[i]
            dJarr *= self.wJ[i, :]
        
        return dh, dJarr
    
    def eval_sequence(self, sequence_str: str, reweight_by_rsa: bool=False) -> np.float32:
        """Evaluate evolutionary energy E of a sequence.
        
        Args:
            sequence_str (str):               AA sequence like 'MAAGKHHIT' of the same length L as the MSA
            reweight_by_rsa (bool=False):     use RSA-complement as weights wh and wJ
        Returns:
            E (np.float32): energy of sequence s
                - E(s) small means energetically favorable sequence
        """
        h, J = self.eval_sequence_hJ(sequence_str, reweight_by_rsa)
        return h + J
    
    def eval_sequence_hJ(self, sequence_str: str, reweight_by_rsa: bool=False) -> Tuple[np.float32, np.float32]:
        """Evaluate 'evolutionary energy' as its single-site and couplings contributions h and J.
        
        arg:
            sequence_str (str)               AA sequence like 'MAAGKHHIT' of the same length as the MSA
            reweight_by_rsa (bool=False)     use RSA-complement as weights wh and wJ
        out:
            * h(s), J(s) small means energetically favorable sequence
        """

        # Convert to int List
        sequence_int = self.encode_sequence_to_ids(sequence_str)

        # Compute energy (minus sign is set so that negative values = stable favorable conformations)
        h_tot, J_tot = np.float32(0.0), np.float32(0.0)

        # Compute (for computational-time optimality, here we just do this ugly large IF/ESLE block)
        # Case: not using RSA-complement weights
        if not reweight_by_rsa:
            for i, aa_i in enumerate(sequence_int):
                h_tot += self.h[i, aa_i]
                for j in range(i):
                    if not self.contacts[i, j]:
                        continue
                    aa_j = sequence_int[j]
                    J_tot += self.J[i, j][aa_i, aa_j]
        # Case: using RSA-complement weights
        else:
            for i, aa_i in enumerate(sequence_int):
                h_tot += self.h[i, aa_i] * self.wh[i]
                for j in range(i):
                    if not self.contacts[i, j]:
                        continue
                    aa_j = sequence_int[j]
                    J_tot += self.J[i, j][aa_i, aa_j] * self.wJ[i, j]
        return -h_tot, -J_tot
    
    def encode_sequence_to_ids(self, sequence_str: str) -> List[int]:
        """Encode a protein sequence (:str) as a list of integers.
        Note: sequence may contain gaps '-' if parameters exclude_gaps=True
        
        arg:
            sequence_str (str)               AA sequence like 'MAAGKHHIT' of the same length as the MSA
        out:
            sequence_int (List[int])         AA sequence encoded in integers (corresponding to StructureDCA amino acid index system)
        """

        # Guardians
        if len(sequence_str) != self.msa_length:
            msg = (
                f"ERROR in DCAModel().encode_sequence_to_ids({format_str(sequence_str)}): \n"
                f" -> length of input sequence ({len(sequence_str)}) does not match length of the MSA ({self.msa_length})"
            )
            raise ValueError(msg)
        if self.exclude_gaps and AminoAcid.GAP_ONE in sequence_str:
            msg = (
                f"ERROR in DCAModel().encode_sequence_to_ids({format_str(sequence_str)}): \n"
                f" -> can not eval a sequence that contains gaps ('{AminoAcid.GAP_ONE}') if parameter exclude_gaps={self.exclude_gaps}"
            )
            raise ValueError(msg)
        for aa in sequence_str:
            if aa not in self.ALL_STATES:
                msg = (
                    f"ERROR in DCAModel().encode_sequence_to_ids({format_str(sequence_str)}): \n"
                    f" -> sequence can not coutain character '{aa}' (allowed characters are '{''.join(self.ALL_STATES)}')"
                )
                raise ValueError(msg)
        
        # Convert to int List
        return self._encode_sequence_to_ids(sequence_str)
    
    @staticmethod
    def _encode_sequence_to_ids(sequence_str: str) -> List[int]:
        """Encode a protein sequence (:str) as a list of integers (no compatibility checks)."""
        return [AminoAcid.ONE_2_ID.get(aa, AminoAcid.GAP_ID) for aa in sequence_str]

    def frobenius_norm(self, apc_correction: bool=False) -> NDArray[np.float32]:
        """Return the matrix F [L x L] of the Frobenius norms between pairs of positions.
        * Measures strength of couplings between positions
        * Gaps coefficients are ignored
        """

        # Init
        L = self.msa_length
        F = np.zeros((L, L), dtype=np.float32)

        # Compute values
        for i in range(L):
            for j in range(i):
                Jij = self.J[i, j]
                Jij_nogaps = Jij[:-1, :-1]
                Fij = np.sqrt(np.sum(Jij_nogaps**2))
                F[i, j] = Fij
                F[j, i] = Fij

        # WARNING: is the APC correction relevent with sparse DCA ???
        # * Average Product Correction (APC):
        #    Ref: S. D. Dunn, L. M. Wahl, and G. B. Gloor, Bioinformatics 24, 333 (2008)
        #    F_APC[i, j] = F[i, j] - F[:, j]*F[i, :]/F[:, :]
        if apc_correction:
            F_line_avg = np.mean(F, axis=0)
            F_avg = np.mean(F)
            F_norm = (F_line_avg[:, None] * F_line_avg[None, :]) / F_avg
            F = F - F_norm
            for i in range(L):
                F[i, i] = 0.0

        # Return
        return F


    # IO Methods ---------------------------------------------------------------
    def export_hyperparameters(self) -> Dict[str, Any]:
        """Export all hyperparameters of the DCAModel (like 'lambda_h' and 'weights_seqid' to a dictionary)."""
        return {
            "msa_path": self.msa_path,                              # input
            "name": self.name,                                      # input property
            "msa_length": self.msa_length,                          # input property
            "msa_depth": self.msa_depth,                            # input property
            "msa_initial_depth": self.msa_initial_depth,            # input property
            "target_sequence": self.target_sequence_str,            # input property
            "lambda_h": self.lambda_h,                              # hyperparameter
            "lambda_J": self.lambda_J,                              # hyperparameter
            "lambda_asymptotic": self.lambda_asymptotic,            # hyperparameter
            "exclude_gaps": self.exclude_gaps,                      # hyperparameter
            "min_seqid": self.min_seqid,                            # hyperparameter
            "weights_seqid": self.weights_seqid,                    # hyperparameter
            "contacts_gap_cutoff": self.contacts_gap_cutoff,        # hyperparameter
            "theta_regularization": self.theta_regularization,      # hyperparameter
            "count_target_sequence": self.count_target_sequence,    # hyperparameter
            "solver": self.solver_name,                             # hyperparameter
            "max_iterations": self.max_iterations,                  # hyperparameter
            "Neff": self.Neff,                                      # property
            "n_contacts": self.n_contacts()                         # property
        }
    
    def write_coeff(self, save_path: str) -> "DCAModel":
        """
        Write DCA coefficients h and J as well as all hyperparameters (msa_path, msa_length, lambda_h, lambda_J, weights_seqid, ...) to a file.
            * Allows to save all h and J coefficients and reload later without having to re-compute the DCA model
            * Uses the NumPy 'np.savez' method and the '.npz' format, so 'save_path' should end with '.npz'
            * Exploit the symmetry in the couplings coefficients J to reduce file size by 2x
            * Exploit contacts to save only non-zero couplings

        Args:
            save_path (str): The file path where the coefficients will be saved. Should end with '.npz'.

        Usage:
            dca_model.write_coeff("./msa_params.npz")
        """

        # Guardians
        save_path = os.path.abspath(save_path)
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        assert save_path.endswith(".npz"), f"ERROR in DCAModel().write_coeff(): save_path='{save_path}' should end with '.npz'."
        assert self.is_initialized, f"ERROR in DCAModel().write_coeff(): DCA model not initialized (there is nothing to save)."

        # Log
        self.logger.step(f"Save DCA coefficients.")
        self.logger.log(f" * dca_cache_file: '{save_path}'")

        # Manage J coefficients
        # Symmetry: J[i1, i2, aa1, aa2] = J[i2, i1, aa2, aa1] and J[i, i, aa1, aa2] = 0
        # Save only non-zero couplings
        L = self.msa_length
        J_compressed = []
        for i in range(L - 1):
            for j in range(i + 1, L):
                if self.contacts[i, j]:
                    Jij = self.J[i, j]
                    J_compressed.append(Jij)
        J_compressed = np.array(J_compressed)

        # Save
        metadata = self.export_hyperparameters()
        metadata = {k: v for k, v in metadata.items() if v is not None} # Avoir bug during reading for None values
        np.savez(
            save_path,                               # path to save '.npz' file
            contacts=self.contacts,                  # Contacts: [L, L]-matrix of couplings to consider
            h=self.h,                                # h coeffs
            J=J_compressed,                          # J coeffs compressed (using symmetry and only non-zero couplings)
            gap_ratios_array=self.gap_ratios_array,  # gap ratios array for each MSA position
            **metadata,
        )
        return self
    
    def read_coeff(self, save_path: str, logger: Union[Logger, bool]=False) -> "DCAModel":
        """
        Read DCA coefficients h and J as well as all hyperparameters (lambda_h, lambda_J, seqid, msa_length, ...) from a file.
            * Allows reload previously saved coefficients without having to re-compute the DCA model
            * Uses the NumPy 'np.savez' method and the '.npz' format, so 'save_path' should end with '.npz'
            * Exploit the symmetry in the couplings coefficients J to reduce file size by 2x
            * Exploit contacts to save only non-zero couplings

        Args:
            save_path (str): Path to the file from which to read the DCA coefficients. Should end with '.npz'.

        Usage:
            dca_model = DCAModel.read_coeff("./msa_param.npz")
        """
        
        # Guardians
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        assert save_path.endswith(".npz"), f"ERROR in DCAModel.read_coeff(): save_path='{save_path}' should end with '.npz'."

        # Log
        logger.step("Read cached DCA model (h and J coefficients).")
        logger.log(f" * dca_cache_path: '{save_path}'")

        # Parse
        dca_data = np.load(save_path, allow_pickle=False)

        # Extract Neff is possible
        if "Neff" in dca_data and dca_data["Neff"] is not None:
            self.Neff = float(dca_data["Neff"])

        # Warning for inconsistent metadata
        current_hyperparameters = self.export_hyperparameters()
        metadata_sequence = dca_data["target_sequence"]
        if self.target_sequence.sequence != metadata_sequence:
            error_log = f"ERROR in DCAModel.read_coeff(): DCA input target sequence from MSA does not match parsed sequences from DCA coefficients metadata:"
            error_log += "\n -> incorrect target sequence from dca_cache_path."
            error_log += f"\n\n * MSA target sequence (from '{self.msa_path}'):   \n{self.target_sequence.sequence}"
            error_log += f"\n\n * DCA coeff cached metadata sequence (from '{save_path}'):   \n{metadata_sequence}"
            raise ValueError(error_log)
        mismatches_arr = []
        for param_name, param_value in current_hyperparameters.items():
            if param_name in ["msa_path", "msa_length", "msa_depth", "msa_initial_depth", "target_sequence"]:
                continue
            param_value_new = None if param_name not in dca_data else dca_data[param_name].item()
            if param_value != param_value_new:
                mismatches_arr.append([param_name, param_value, param_value_new])
        if len(mismatches_arr) > 0:
            warning_log = f"some input parameters of StructureDCA do not match extracted metadata from DCA coefficients:"
            warning_log += "\n  -> DCA coefficients were computed using parameters from cached file and not the one you give as input."
            for param_name, param_value, param_value_new in mismatches_arr:
                warning_log += f"\n     * {param_name}: {param_value} (input) != {param_value_new} (cached file)"
            self.logger.warning(warning_log, critical=True)

        # Assing contacts
        contacts_input = self.contacts
        self.contacts = dca_data["contacts"]

        # Non matching contacts WARNING
        if not np.array_equal(contacts_input, self.contacts):
            L = self.msa_length
            n_input_not_cache, n_cache_not_input = 0, 0
            for i in range(L):
                for j in range(L):
                    input_c = contacts_input[i, j]
                    cache_c = self.contacts[i, j]
                    if input_c != cache_c:
                        n_input_not_cache += int(input_c)
                        n_cache_not_input += int(cache_c)
            warning_log = f"input DCA contacts do not match with cached contacts."
            warning_log += f"\n  -> {n_input_not_cache} / {L*L} couplings in input but not in cache"
            warning_log += f"\n  -> {n_cache_not_input} / {L*L} in cache but not in input"
            self.logger.warning(warning_log, critical=True)

        # Copy MSA depth metadata if possible
        if "msa_depth" in dca_data:
            self.msa_depth = dca_data["msa_depth"].item()
        if "msa_initial_depth" in dca_data:
            self.msa_initial_depth = dca_data["msa_initial_depth"].item()

        # Assign gap ratios
        self.gap_ratios_array = dca_data["gap_ratios_array"]
        
        # Assign DCA model: h
        self.h = dca_data["h"]

        # Assign J by unflatten saved compressed J_sparse (using symmetry and saving only non-zero couplings)
        L = self.msa_length
        A = self.N_STATES
        J_compressed = dca_data["J"]
        del dca_data # free cache

        # Read J
        if self.use_sparse_J:
            J = SparseJ(L, A, logger=self.logger)
            set_symmetrically = SparseJ.set_symmetrically_sparse
        else:
            J = np.zeros((L, L, A, A), dtype=np.float32)
            set_symmetrically = SparseJ.set_symmetrically_full
        coupling_index = 0
        for i in range(L - 1):
            for j in range(i + 1, L):
                if not self.contacts[i, j]:
                    continue
                set_symmetrically(J, i, j, J_compressed[coupling_index])
                coupling_index += 1

        # Return
        self.J = J
        self.is_initialized = True
        return self
