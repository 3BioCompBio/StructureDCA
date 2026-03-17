
# Imports ----------------------------------------------------------------------
import os.path
from os import cpu_count
from typing import Union, Literal, List, Dict
import numpy as np
from numpy.typing import NDArray
from littlecsv import CSV
from structuredca.utils import Logger, format_str
from structuredca.sequence import Sequence, FastaReader, PairwiseAlignment, AminoAcid
from structuredca.structure import Structure
from structuredca.aligner import StructureSequenceAlignment
from structuredca.dca_model import DCAModel
from structuredca.dca_model.dca_solvers import DCASolver
from structuredca.dca_model.data_structures import SparseJ


# Main -------------------------------------------------------------------------
class StructureDCA:
    """StructureDCA: Structure-informed DCA model of an MSA."""


    # Constructors -------------------------------------------------------------
    def __init__(
            self,
            msa_path: str,
            pdb_path: Union[str, None],
            chains: str="A",
            homomeric_chains: Union[None, str]=None,
            distance_cutoff: Union[None, float]=8.00,
            lambda_h: float=1.00,
            lambda_J: float=1.00,
            lambda_asymptotic: float=0.001,
            exclude_gaps: bool=True,
            min_seqid: Union[None, float]=0.25,
            weights_seqid: Union[None, float]=0.80,
            use_contacts_plddt_filter: bool=False,
            contacts_plddt_cutoff: float=70.0,
            contacts_plddt_keep_window: int=1,
            contacts_gap_cutoff: Union[None, float]=None,
            theta_regularization: float=0.10,
            count_target_sequence: bool=True,
            ignore_hydrogen_atoms: bool=True,
            ignore_backbone_atoms: bool=True,
            num_threads: int=4,
            solver: Union[Literal["plmDCA"], DCASolver]="plmDCA",
            max_iterations: int=2000,
            use_sparse_J: bool=True,
            init_dca=True,
            distance_cache_path: Union[None, str]=None,
            rsa_cache_path: Union[None, str]=None,
            weights_cache_path: Union[None, str]=None,
            dca_cache_path: Union[None, str]=None,
            verbose: bool=True,
            log_gd_steps: bool=False,
            disable_warnings: bool=False,
        ):
        
        """StructureDCA: Structure-Informed DCA model of an MSA.

-------------------------------------------------------------------------------
usage (Python):
   from structuredca import StructureDCA                                   # Import pip package
   sdca = StructureDCA('./msa1.fasta', './pdb1.pdb', 'A')                  # Initialize StructureDCA with an MSA, a PDB and the corresponding chain(s) in the PDB
   mutation_score = sdca.eval_mutation('K24M:H39G', reweight_by_rsa=False) # Compute the score of a missense single/multiple mutation
   all_muts_scores_table = sdca.eval_mutations_table()                     # Get scores for all single-site missense mutations

-------------------------------------------------------------------------------
Input arguments:
   msa_path (str)                              path to MSA '.fasta', '.a2m' or '.a3m' file (can be zipped with '.gz')
   pdb_path (str)                              path to PDB '.pdb' file
   chains (str)                                target chain(s) in the PDB to consider (that corresponds to the target chain(s) in the MSA)
   homomeric_chains (None | str, None)         homomer groups to consider inter-chains contacts (e.g. 'ABC' for homo-trimer or 'AC:BD' for two homo-dimers)
    
StructureDCA arguments:
   distance_cutoff (None | float, 8.0)         distance threshold (in Å) to consider a residue-residue contact
   lambda_h (float, 1.0)                       L2 regularization for h
   lambda_J (float, 1.0)                       L2 regularization for J
   lambda_asymptotic (float, 0.001)            L2 regularization asymptotic correction (when Neff -> +inf)
   exclude_gaps (bool, True)                   exclude gaps from the DCA model (if False, consider gap as the 21th amino acid)
   min_seqid (None | float, 0.25)              sequences which sequence-identity with target sequence is below this will be discarded (set None to ignore)
   weights_seqid (None | float, 0.80)          sequence-identity threshold for sequences weighting (set None to ignore)
   use_contacts_plddt_filter (bool, False)     remove contacts at positions where pLDDT is lower than a given threshold in predicted structures
   contacts_plddt_cutoff (float, 70.0)         pLDDT threshold below which contacts are discarded
   contacts_plddt_keep_window (int, 1)         if contacts_plddt_cutoff is used: size of the position window where contacts falling below the pLDDT threshold are still kept
   contacts_gap_cutoff (None | float, None)    remove contacts at positions where gap-ratio is higher than this threshold
   theta_regularization (float, 0.10)          regularization at frequency level (only for initialization of DCA coefficients h)
   count_target_sequence (bool, True)          count target (first) sequence of the MSA for the DCA model

Structure arguments:
   ignore_hydrogen_atoms (bool, True)          ignore hydrogen atoms to compute res-res distances 
   ignore_backbone_atoms (bool, True)          ignore backbone atoms to compute res-res distances 

Execution arguments:
   num_threads (int, 4)                        number of threads (CPUs) for DCA solver
   solver (str, 'plmDCA')                      DCA solver among ['plmDCA']
   max_iterations (int, 2000)                  maximum number of GD iterations to solve DCA model
   use_sparse_J (bool, True)                   specify if use a sparse implementation or a usual numpy.ndarray for couplings coefficients J
   init_dca (bool, True)                       initialize DCA model 

Cache arguments:
   distance_cache_path (None | str, None)      path to write/read to/from res-res distances file (should be a '.npy' file)
   rsa_cache_path (None | str, None)           path to write/read to/from RSA values
   weights_cache_path (None | str, None)       path to write/read to/from MSA sequences weights
   dca_cache_path (None | str, None)           path to write/read to/from DCA parameters and coefficients h and J

Logging arguments:
   verbose (bool, True)                        log execution steps
   log_gd_steps (bool, False)                  log gradient descent steps for plmDCA
   disable_warnings (bool, False)              disable logging for Warnings (use with caution)

-------------------------------------------------------------------------------
StructureDCA Properties:
   sdca.msa_length [L] (int)                           aa-length of the target sequence of the MSA
   sdca.msa_depth [N] (int)                            number of sequences in the MSA
   sdca.n_states [q] (int)                             21, number of possible states (20 AAs and 1 gap)
   sdca.h (np.array[float32] (L, q))                   DCA fields coefficients h
   sdca.J (SparseJ[float32] (L, L, q, q))              DCA couplings coefficients J (access with J[i, j] and J[i, j][a, b] but not J[i, j, a, b])
   sdca.alignment (PairwiseAlignment)                  alignment between target chain(s) in PDB and target sequence in MSA
   sdca.distance_matrix (np.array[float32] (L, L))     res-res distance matrix
   sdca.contacts (np.array[bool] (L, L))               res-res contact matrix
   sdca.rsa_array (np.array[float32] (L))              array of RSA values for each residue / MSA position
   sdca.plddt_array (np.array[float32] (L))            array of pDLLT (or B-factor) values for each residue / MSA position
   sdca.gap_ratios_array (np.array[float32] (L))       array of gap ratios values for each MSA position
        """

        # Guardian, Logger and base properties
        self.name = FastaReader.filename(msa_path)
        self.logger = Logger(verbose, disable_warnings, step_prefix="StructureDCA", warning_note=f" in {self}")
        self.log_gd_steps = log_gd_steps
        self.dca_cache_path = dca_cache_path
        self.use_contacts_plddt_filter = use_contacts_plddt_filter
        self.contacts_plddt_cutoff = contacts_plddt_cutoff
        self.contacts_plddt_keep_window = contacts_plddt_keep_window
        self.distance_cutoff = distance_cutoff
        if self.logger.verbose:
            print("\n--------------------------------------------------------------------------------")
        self.logger.step(f"Initialize {self}.")

        # Early guard for coefficients
        if weights_cache_path is not None: 
            assert weights_cache_path.endswith(".npz"), f"ERROR in StructureDCA: weights_cache_path='{weights_cache_path}' should end with '.npz'."

        # Init target sequence (first sequence in the MSA)
        self.logger.step("Read MSA target sequence (first sequence in the MSA).")
        self.target_sequence: Sequence = FastaReader.read_first_sequence(msa_path)
        self.logger.log(f" * sequence: '{self.target_sequence.seq_short(30)}' (l={self.L})")

        # Init structure
        if pdb_path is None: # Case: create a decoy fully-connected structure from target_sequence
            pdb_path = self.target_sequence
        self.structure = Structure(
            pdb_path,
            chains,
            homomeric_chains=homomeric_chains,
            ignore_hydrogen_atoms=ignore_hydrogen_atoms,
            ignore_backbone_atoms=ignore_backbone_atoms,
            solve_rsa=True,
            solve_distances=True,
            distance_cache_path=distance_cache_path,
            rsa_cache_path=rsa_cache_path,
            logger=self.logger,
        )

        # Too much CPU warning
        if self.L < num_threads:
            self.logger.warning(f"num_threads={num_threads} exeeds MSA target sequence length {self.L}: value was adapted: {num_threads} -> {self.L}.")
            num_threads = self.L
        num_cpu_total = cpu_count()
        if num_cpu_total is None or num_cpu_total < num_threads:
            self.logger.warning(f"num_threads={num_threads} exeeds total number of CPUs detected on current machine (num_cpu_total={num_cpu_total}).")
        
        # Init structure - sequence aligner
        self.structure_sequence_alignment = StructureSequenceAlignment(
            self.structure,
            self.target_sequence,
            logger=self.logger,
        )

        # Init contacts based on distance_cutoff and pLDDT
        n_contacts_max = self.L**2 - self.L # all pairs of residues counting twice (i, j) and (j, i) and except contacts with self (i, i)
        contacts: NDArray[np.bool_]
        if self.distance_cutoff is None:    # Set that all non-self couplings are contacts
            contacts = np.ones((self.L, self.L), np.bool_)
            for i in range(self.L):         # Disable contacts at diagonal
                contacts[i, i] = False
            n_contacts = n_contacts_max
        else:                               # Set contacts for which d(i, j) < d_thr as contacts
            if self.distance_cutoff < 0.0:
                raise ValueError(f"ERROR in {self}: distance_cutoff={distance_cutoff} schould be positive.")
            self.logger.step(f"Set distance-based contacts matrix [{self.L}x{self.L}] (for sparse DCA).")
            self.logger.log(f" * distance_cutoff: {distance_cutoff:.2f}")
            plddt_filter = self.get_plddt_filter()
            contacts = (self.distance_matrix < distance_cutoff) & plddt_filter
            for i in range(self.L):         # Disable contacts at diagonal
                contacts[i, i] = False
            n_contacts = int(np.sum(contacts))
            self.logger.log(f" * remaining couplings: {n_contacts} / {n_contacts_max} ({100.0*n_contacts / (n_contacts_max):.2f} %)")
        self.logger.log(f" * average contacts by position: {n_contacts / self.L:.2f}")

        # Initialize weights wh and wJ using RSA-complement
        wh = 1.0 - np.minimum(self.rsa_array, 100.0) / 100.0 # h_weights (L): (1 - RSAc[i])
        wJ = (wh[:, None] + wh[None, :]) / 2.0 # J_weights (L, L): ((1 - RSAc[i]) + (1 - RSAc[j])) / 2

        # Initialize DCA model (h and J coefficients)
        self.logger.step(f"Initialize DCA solver.")
        self.dca_model = DCAModel(
            msa_path=msa_path,
            name=self.name,
            contacts=contacts,
            wh=wh,
            wJ=wJ,
            lambda_h=lambda_h,
            lambda_J=lambda_J,
            lambda_asymptotic=lambda_asymptotic,
            exclude_gaps=exclude_gaps,
            min_seqid=min_seqid,
            weights_seqid=weights_seqid,
            contacts_gap_cutoff=contacts_gap_cutoff,
            theta_regularization=theta_regularization,
            count_target_sequence=count_target_sequence,
            num_threads=num_threads,
            solver=solver,
            max_iterations=max_iterations,
            use_sparse_J=use_sparse_J,
            weights_cache_path=weights_cache_path,
            logger=self.logger,
            log_gd_steps=self.log_gd_steps,
        )
        # Assign Energy evaluation methods DCAModel directly to StructureDCA
        self.n_contacts = self.dca_model.n_contacts
        self.avg_contacts = self.dca_model.avg_contacts
        self.eval_mutation = self.dca_model.eval_mutation
        self.position_probabilities = self.dca_model.position_probabilities
        self.eval_mutation_hJ = self.dca_model.eval_mutation_hJ
        self.eval_mutation_hJarr = self.dca_model.eval_mutation_hJarr
        self.eval_sequence = self.dca_model.eval_sequence
        self.eval_sequence_hJ = self.dca_model.eval_sequence_hJ


        # Init DCA unless specified otherwise
        if init_dca:
            self.init_dca()
        else:
            msg = (
                "DCA initialization was prevented by parameter init_dca=False \n"
                " -> please run sdca.init_dca() before using the DCA model"
            )
            self.logger.warning(msg)

    def init_dca(self) -> "StructureDCA":
        """StructureDCA.init_dca(): Initialize / Solve DCA model: h, J coefficients.
        - If dca_cache_path is not None and file already exists, read DCA coefficients from file
        - If dca_cache_path is not None and file does not exists, write DCA coefficients to file
        """

        # Case: DCA already initialized
        if self.dca_model.is_initialized:
            self.logger.warning("DCA model already initialized -> StructureDCA.init_dca() was skipped.")
            return self

        # Case: DCA coefficients are saved in cache
        if self.dca_cache_path is not None and os.path.isfile(self.dca_cache_path):
            self.dca_model.read_coeff(self.dca_cache_path, logger=self.logger)
            return self
        
        # Base Case: Solve DCA model
        self.dca_model.init_dca()
        if self.dca_cache_path is not None:
            self.dca_model.write_coeff(self.dca_cache_path)
        return self


    # Evaluate Mutations Scores (Evolutionary Energy) --------------------------
    def eval_mutations_table(
            self,
            mutations: Union[None, List[str]]=None,
            as_in_pdb: bool=False,
            background_sequence: Union[None, str]=None,
            save_path: Union[None, str]=None,
            round_digit: Union[None, int]=6,
            sep: str=",",
            log_output_sample: bool=False,
            n_output_sample_lines: int=10,
        ) -> List[Dict[str, Union[str, float]]]:
        """
        Evaluate DCA score (Evolutionary Delta Energy, 'dE') for a list of mutations.
            -> dE > 0 means unfavorable / destabilizing mutation
            -> if 'mutations' is not specified, it will predict all single-site missense mutations
        
        Parameters:
            mutations (List[str]): List of mutations like: ['M13G', 'M13G:H15k'].
            as_in_pdb (bool, False): Set True to provide mutations with respect to ATOM lines in PDB file (like 'MA12G' instead of 'M13G').
            background_sequence (str=None): alternative target sequence to mutate from
            save_path (str, None): Path to a '.csv' file to save results.
            round_digit (int, 6): Parameter to round all numerical values.
            sep (str, ','): Separator of the '.csv' file.
            log_output_sample (bool, False): Log a sample of the generated output.
            n_output_sample_lines (int, 10): if log_output_sample, how many lines are logged
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries for each mutation containing properties
            - 'mutation_fasta': mutation as represented in the fasta format (first residue is 1 and then increment)
            - 'mutation_pdb': mutation as represented in the ATOM lines of the PDB
            - 'StructureDCA': dE, delta evolutionary energy DCA score (dE = dh + dJ) of mutation
            - 'RSA*StructureDCA': dE, delta evolutionary energy DCA score (dE = dh + dJ) of mutation, reweighted by RSA
            - 'RSA': RSA of mutated residue(s) (separated by ':')
            - 'pLDDT': pLDDT of mutated residue(s) (separated by ':')
            - 'gap_ratio': gap_ratio of mutated residue(s) (separated by ':')
            - 'warnings': warnings tags if there are some (separated by ':')
        """

        # Save path guardians
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            assert save_path.endswith(".csv"), f"ERROR in {self}: in sdca.eval_mutations_table(): save_path='{save_path}' should be a '.csv' file."

            # Create output directory if it does not exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Set background sequence to self.target_sequence if it is not specified
        if background_sequence is None:
            target_sequence = self.target_sequence
        else:
            target_sequence = Sequence("custom_background_sequence", background_sequence)

        # Sep guadiants
        assert sep != ":", f"ERROR in {self}: sep=':' is forbidden as it is the separator for multiple mutations."

        # Case: all single mutations DMS
        if mutations is None:
            mutations = target_sequence.list_all_single_mutations()
            as_in_pdb = False

        # Log
        self.logger.step(f"Evaluate DCA DeltaE scores for {len(mutations)} mutations.")

        # Translate mutations from PDB to Fasta conventions
        if as_in_pdb:
            self.logger.log(f" * convert mutations from PDB to Fasta format")
            mutations_as_fasta = []
            for mutation in mutations:
                try:
                    mutation_as_fasta = self.structure_sequence_alignment.convert_to_sequence_mutation(mutation)
                    mutations_as_fasta.append(mutation_as_fasta)
                except:
                    continue
            if len(mutations_as_fasta) == 0:
                raise ValueError(f"ERROR in {self}: in sdca.eval_mutations_table({format_str(str(mutations))}): all mutations failed to convert from PDB to Fasta format.")
            n_error = len(mutations) - len(mutations_as_fasta)
            if n_error > 0:
                self.logger.warning(f".eval_mutations_table({format_str(str(mutations))}): {n_error} / {len(mutations)} mutations failed to convert from PDB to Fasta format.")
            mutations = mutations_as_fasta

        # Guardian
        if not (isinstance(mutations, list) and len(mutations) > 0 and all([isinstance(m, str) for m in mutations])):
            correction_log = " -> input 'mutations' should be a non-empty list of mutations as strings (like ['M13G', 'M13G:H15k'])."
            raise ValueError(f"ERROR in {self}: in sdca.eval_mutations_table({format_str(str(mutations))}): invalid input mutations: \n{correction_log}")

        # Evalutate mutations list
        scores: List[Dict[str, Union[str, float]]] = []
        for mutation in mutations:
            mut_positions = [int(mut[1:-1])-1 for mut in mutation.split(":")]
            rsa_arr = [round(float(self.rsa_array[i]), 2) for i in mut_positions]
            plddt_arr = [round(float(self.plddt_array[i]), 2) for i in mut_positions]
            gap_ratios_arr = [round(float(self.dca_model.gap_ratios_array[i]), 2) for i in mut_positions]
            warnings = ":".join(self.structure_sequence_alignment.get_alignment_warnings_list(mut_positions))
            dE = self.eval_mutation(mutation, reweight_by_rsa=False, background_sequence=background_sequence)
            dE_rsa = self.eval_mutation(mutation, reweight_by_rsa=True, background_sequence=background_sequence)
            try:
                mutation_pdb = self.structure_sequence_alignment.convert_to_structure_mutation(mutation)
            except:
                mutation_pdb = ""
            mutation_scores = {
                "mutation_fasta": mutation,
                "mutation_pdb": mutation_pdb,
                "StructureDCA": float(dE),
                "RSA*StructureDCA": float(dE_rsa),
                "RSA": rsa_arr,
                "pLDDT": plddt_arr,
                "gap_ratio": gap_ratios_arr,
                "warnings": warnings,
            }
            scores.append(mutation_scores)

        # Round float values if required
        if round_digit is not None:
            for score in scores:
                for prop in ["StructureDCA", "RSA*StructureDCA"]:
                    score[prop] = round(score[prop], round_digit)

        # Format in a CSV dataframe
        if log_output_sample or save_path is not None:
            scores_csv = CSV(list(scores[0].keys()), name=self.name, sep=sep)
            for score_entry in scores:
                score_entry_copy = {k: v for k, v in score_entry.items()}
                for prop in ["RSA", "pLDDT", "gap_ratio"]: # post-process properties that may be lists before saving to .csv file
                    score_entry_copy[prop] = ":".join([f"{x:.2f}" for x in score_entry_copy[prop]])
                scores_csv.add_entry(score_entry_copy)

        # Log results
        if log_output_sample:
            round_digit_log = round_digit if round_digit is not None else 6
            scores_csv.show(n_entries=n_output_sample_lines, max_col_length=20, round_digit=round_digit_log)

        # Save
        if save_path is not None:
            self.logger.log(f" * save scores to: '{save_path}'")
            scores_csv.write(save_path)

        # Return
        return scores
    
    def eval_positions_table(
            self,
            save_path: Union[None, str]=None,
            round_digit: Union[None, int]=6,
            sep: str=",",
            log_output_sample: bool=False,
            n_output_sample_lines: int=10,
        ) -> List[Dict[str, Union[str, float]]]:
        """
        Evaluate DCA average positional score (Evolutionary Delta Energy, 'dE') for all MSA positions.
            -> dE > 0 means unfavorable / destabilizing positions to mutate
        
        Parameters:
            save_path (str, None): Path to a '.csv' file to save results.
            round_digit (int, 6): Parameter to round all numerical values.
            sep (str, ','): Separator of the '.csv' file.
            log_output_sample (bool, False): Log a sample of the generated output.
            n_output_sample_lines (int, 10): if log_output_sample, how many lines are logged
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries for each mutation containing properties
            - 'resid_fasta': residue ID as represented in the fasta format (like 'K16')
            - 'resid_pdb': residue ID as represented in the ATOM lines of the PDB (like 'KA16' where 'A' is the chain)
            - 'StructureDCA': average positional dE, delta evolutionary energy DCA score (dE = dh + dJ) of mutation
            - 'RSA*StructureDCA': average positional dE, delta evolutionary energy DCA score (dE = dh + dJ) of mutation, reweighted by RSA
            - 'RSA': RSA of residue
            - 'pLDDT': pLDDT of residue
            - 'gap_ratio': gap_ratio of residue
            - 'warning': warning tag if there is one
        """

        # Save path guardians
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            assert save_path.endswith(".csv"), f"ERROR in {self}: in sdca.eval_positions_table(): save_path='{save_path}' should be a '.csv' file."

        # Create output directory if it does not exist
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Log
        self.logger.step(f"Evaluate DCA DeltaE positional scores for {self.L} positions.")

        # Evalutate mutations list
        aas = [aa.one for aa in AminoAcid.get_all()]
        scores: List[Dict[str, Union[str, float]]] = []
        for i, wt_aa in enumerate(self.target_sequence.sequence):
            rsa = self.rsa_array[i]
            plddt = self.plddt_array[i]
            gap_ratio = self.dca_model.gap_ratios_array[i]
            warning = self.structure_sequence_alignment.get_alignment_warning(i)
            warning_str = "" if warning is None else warning
            positionid_fasta = str(i+1)
            dE = np.mean([
                self.eval_mutation(f"{wt_aa}{positionid_fasta}{mt_aa}", reweight_by_rsa=False)
                for mt_aa in aas
            ])
            dE_rsa = np.mean([
                self.eval_mutation(f"{wt_aa}{positionid_fasta}{mt_aa}", reweight_by_rsa=True)
                for mt_aa in aas
            ])
            resid_fasta = f"{wt_aa}{positionid_fasta}"
            positionid_pdb = self.structure_sequence_alignment.seq_2_str_residues.get(positionid_fasta)
            if positionid_pdb is not None:
                resid_pdb = wt_aa + positionid_pdb
            else:
                resid_pdb = ""
            mutation_scores = {
                "resid_fasta": resid_fasta,
                "resid_pdb": resid_pdb,
                "StructureDCA": round(float(dE), round_digit),
                "RSA*StructureDCA": round(float(dE_rsa), round_digit),
                "RSA": round(rsa, 2),
                "pLDDT": round(plddt, 2),
                "gap_ratio": round(gap_ratio, 2),
                "warning": warning_str,
            }
            scores.append(mutation_scores)

        # Format in a CSV dataframe (positions)
        if log_output_sample or save_path is not None:
            scores_csv = CSV(list(scores[0].keys()), name=self.name, sep=sep)
            for score_entry in scores:
                scores_csv.add_entry(score_entry)

        # Log results (positions)
        if log_output_sample:
            round_digit_log = round_digit if round_digit is not None else 6
            scores_csv.show(n_entries=n_output_sample_lines, max_col_length=20, round_digit=round_digit_log)

        # Save (positions)
        if save_path is not None:
            self.logger.log(f" * save scores to: '{save_path}'")
            scores_csv.write(save_path)

        # Return (positions)
        return scores

    # Properties ---------------------------------------------------------------
    @classmethod
    def help(cls) -> None:
        """Log main usage (help) of StructureDCA."""

        # Get docstring
        doc_str = cls.__init__.__doc__

        # Title
        title = "StructureDCA: Structure-informed DCA model of an MSA."
        doc_str = doc_str.replace(title, f"{Logger.OKGREEN}{Logger.BOLD}{title}{Logger.ENDC}")

        # Color important arguments
        important_arguments_list = [
            "msa_path", "pdb_path", "chains",
            "distance_cutoff", "lambda_h", "lambda_J",
            "exclude_gaps", "min_seqid", "weights_seqid", "use_contacts_plddt_filter",
            "num_threads", "verbose",
        ]
        for important_argument in important_arguments_list:
            important_argument_in  = f"  {important_argument} "
            important_argument_out = f"  {Logger.OKGREEN}{important_argument}{Logger.ENDC} "
            doc_str = doc_str.replace(important_argument_in, important_argument_out)

        # Bold subtitles
        subtitles_list = [
            "StructureDCA: Structure-Informed DCA model of an MSA.",
            "usage (Python)",
            "Input arguments", "StructureDCA arguments", "Structure arguments", "Execution arguments",
            "Cache arguments", "Logging arguments",
            "StructureDCA Properties",
        ]
        for subtitle in subtitles_list:
            subtitle_in  = f"{subtitle}"
            subtitle_out = f"{Logger.BOLD}{subtitle}{Logger.ENDC}"
            doc_str = doc_str.replace(subtitle_in, subtitle_out, 1)

        # Log
        print(doc_str)

    def __str__(self) -> str:
        return f"StructureDCA('{self.name}')"
        
    @property
    def msa_length(self) -> int:
        """MSA length -> length of target sequence (aliases 'msa_length', 'L')"""
        return len(self.target_sequence)
    
    @property
    def L(self) -> int:
        """MSA length -> length of target sequence (aliases 'msa_length', 'L')"""
        return len(self.target_sequence)
    
    @property
    def msa_depth(self) -> int:
        """MSA depth -> number of sequences in MSA (aliases 'msa_depth', 'N')"""
        return self.dca_model.msa_depth
    
    @property
    def N(self) -> int:
        """MSA depth -> number of sequences in MSA (aliases 'msa_depth', 'N')"""
        return self.dca_model.msa_depth
    
    @property
    def n_states(self) -> int:
        "21, number of possible states (20 aas and 1 gap) (aliases 'n_states', 'q')"
        return self.dca_model.N_STATES
    
    @property
    def q(self) -> int:
        "21, number of possible states (20 aas and 1 gap) (aliases 'n_states', 'q')"
        return self.dca_model.N_STATES
    
    @property
    def h(self) -> NDArray[np.float32]:
        """DCA fields coefficients h (L, q)"""
        return self.dca_model.h
    
    @property
    def J(self) -> SparseJ:
        """DCA couplings coefficients J (L, L, q, q)"""
        return self.dca_model.J
    
    @property
    def alignment(self) -> PairwiseAlignment:
        """Pairwise aligmnent between structure and sequence"""
        return self.structure_sequence_alignment.align

    @property
    def distance_matrix(self) -> NDArray[np.float32]:
        """Distance (L, L)-matrix on residues positions pairs in MSA target sequence"""
        return self.structure_sequence_alignment.distance_matrix
    
    @property
    def contacts(self) -> NDArray[np.bool_]:
        """Contacts (L, L)-matrix between residues positions pairs in MSA target sequence"""
        return self.dca_model.contacts

    @contacts.setter
    def contacts(self, contacts: NDArray[np.bool_]):
        self.dca_model.contacts = contacts
    
    @property
    def rsa_array(self) -> NDArray[np.float32]:
        """RSA (L)-array on residues positions in MSA target sequence"""
        return self.structure_sequence_alignment.rsa_array
    
    @property
    def plddt_array(self) -> NDArray[np.float32]:
        """pLDDT (or B-factor) (L)-array on residues positions in MSA target sequence"""
        return self.structure_sequence_alignment.plddt_array
    
    @property
    def gap_ratios_array(self) -> NDArray[np.float32]:
        """Gap ratios (L)-array on positions in MSA"""
        return self.dca_model.gap_ratios_array

    # Dependencies -------------------------------------------------------------
    def get_plddt_filter(self) -> NDArray[np.bool_]:
        """Returns a (L, L)-matrix where (i, j) is set to:
            - False when pLDDT at position i or j is < contacts_plddt_cutoff
            - except in a diagonal window where |i-j| <= contacts_plddt_keep_window
        """

        # Prevent pLDDT filter if 3D Structure is detected as experimental
        if self.use_contacts_plddt_filter:
            if self.structure.is_experimental():
                self.logger.warning(f"{self.structure} is detected to be experimental (by its 'EXPDTA' line) -> pLDDT filter {self.contacts_plddt_cutoff} is DISABLED.")
                self.use_contacts_plddt_filter = False
            elif self.structure.is_probably_experimental():
                self.logger.warning(f"{self.structure} is detected to be probably experimental (by its RSA vs. B-factor correlation)")
                self.logger.warning(f" -> if it is the case, you should disable pLDDT filter (currently use_contacts_plddt_filter='{self.use_contacts_plddt_filter}')")

        # Init
        p = self.contacts_plddt_cutoff
        w = self.contacts_plddt_keep_window
        L = self.L

        # Base case: if pLDDT filter is trivial
        if not self.use_contacts_plddt_filter or p <= 0.0:
            return np.ones((L, L), dtype=np.bool_)

        # Guardians
        assert p is not None, f"ERROR in {self}: use_contacts_plddt_filter is True but contacts_plddt_cutoff is None. If you want to disable pLDDT filter, set use_contacts_plddt_filter to False."
        assert 0.0 <= p <= 100.0, f"ERROR in {self}: contacts_plddt_cutoff={p} should be in [0, 100]."
        assert isinstance(w, int), f"ERROR in {self}: contacts_plddt_keep_window={w} should be an integer."
        assert w >= 0, f"ERROR in {self}: contacts_plddt_keep_window={w} should be positive."
        
        # Warning: contacts_plddt_cutoff is suspicious
        if 0.0 < p <= 1.0:
            self.logger.warning(f"suspicious value for contacts_plddt_cutoff={p:.1f}: pLDDT range is [0.0, 100.0] -> do you mean contacts_plddt_cutoff={p*100.0} ?")
        
        # Compute pLDDT filter
        plddt_filter_array = self.plddt_array >= p
        plddt_filter = np.outer(plddt_filter_array, plddt_filter_array)
        diag_filter = np.triu(np.tril(np.ones((L, L), dtype=np.bool_), w), -w)
        plddt_filter = plddt_filter | diag_filter

        # Log and return
        n_discarded_residues = L - np.sum(plddt_filter_array)
        self.logger.log(f" * pLDDT filter (plddt_cutoff={p}, plddt_keep_window={w}): {n_discarded_residues} / {L} discarded residues")
        return plddt_filter
