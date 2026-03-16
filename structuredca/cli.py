
# Imports ----------------------------------------------------------------------
import os.path
import argparse
from structuredca import StructureDCA

from structuredca.sequence.fasta_reader import FastaStream
from structuredca.sequence.mutation import read_mutations_file

# CLI: dependencies ------------------------------------------------------------
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.HelpFormatter):

    # Increase max_help_position
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, max_help_position=40)

    # Overwrite help string to automatically write (default: ...) when not None
    def _get_help_string(self, action):
        help_text = action.help

        if action.default is None or isinstance(action.default, bool):
            return help_text

        return super()._get_help_string(action)

    # Only show metavar once 
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)

        parts = action.option_strings[:]
        if action.nargs != 0:
            parts[-1] += f" {self._format_args(action, action.dest)}"
        return ", ".join(parts)


# CLI: main function -----------------------------------------------------------
def main():
    

    # Define Argument Parser ---------------------------------------------------

    # Init parser
    parser = argparse.ArgumentParser(
        description="Run StructureDCA to evaluate effects of mutations on a target protein.",
        usage="structuredca <msa_path> <pdb_path> <chains> [options]\nhelp:  structuredca -h",
        formatter_class=CustomHelpFormatter,
    )

    parser.add_argument(
        "msa_path", type=str,
        help=f"path to MSA file ({', '.join(FastaStream.ACCEPTED_MSA_EXTENSIONS)})",
    )

    parser.add_argument(
        "pdb_path", type=str,
        help="path to PDB '.pdb' file",
    )

    parser.add_argument(
        "chains", type=str,
        help="chain(s) in the PDB to consider",
    )

    parser.add_argument(
        "-o", "--output_path", type=str, default=None, metavar="<str>",
        help="path to output '.csv' file (default: './[msa_filename]_structuredca.csv')",
    )

    parser.add_argument(
        "-i", "--mfile", type=str, default=None, metavar="<str>",
        help="path to a mutations file (if not provided, run all single-site mutations): one line by mutation (like 'A54H' or 'M13A:G15K')"
    )

    parser.add_argument(
        "-d", "--distance_cutoff", type=float, default=8.00, metavar="<float>",
        help="distance threshold (in Å) to consider a residue-residue contact",
    )

    parser.add_argument(
        "-lh", "--lambda_h", type=float, default=1.00, metavar="<float>",
        help="L2 regularization for h",
    )

    parser.add_argument(
        "-lJ", "--lambda_J", type=float, default=1.00, metavar="<float>",
        help="L2 regularization for J",
    )

    parser.add_argument(
        "-g", "--model_gaps", dest="exclude_gaps", action="store_false",
        help="consider gap symbols in the DCA model as any other amino acid",
    )

    parser.set_defaults(exclude_gaps=True)

    parser.add_argument(
        "-m", "--min_seqid", type=float, default=0.25, metavar="<float>",
        help="discard sequences which seqid with target sequence is below",
    )

    parser.add_argument(
        "-w", "--weights_seqid", type=float, default=0.80, metavar="<float>",
        help="seqid threshold for weighting"
    )

    parser.add_argument(
        "--plddt", nargs="?", const=70.0, type=float, metavar="<float>",
        help="enable pLDDT filtering, optionally set threshold (70.0 by default)"
    )

    parser.add_argument(
        "-t", "--theta_regularization", type=float, default=0.10, metavar="<float>",
        help="regularization at frequency level (only for initialization of DCA coefficients h)",
    )

    parser.add_argument(
        "--ignore_target_sequence", dest="count_target_sequence", action="store_false",
        help="ignore target (first) sequence of the MSA in frequencies",
    )
    parser.set_defaults(count_target_sequence=True)

    parser.add_argument(
        "-N", "--num_threads", type=int, default=4, metavar="<int>",
        help="number of threads (CPUs) for DCA solver",
    )

    parser.add_argument(
        "-S", "--silent", dest="verbose", action="store_false",
        help="run in silent mode",
    )
    parser.set_defaults(verbose=True)

    parser.add_argument(
        "-W", "--disable_warnings", dest="disable_warnings", action="store_true",
        help="disable logging of warnings",
    )
    parser.set_defaults(disable_warnings=False)

    parser.add_argument(
        "--sep", type=str, default=";", metavar="<str>",
        help="separator in the output '.csv' file",
    )

    parser.add_argument(
        "--weights_cache_path", type=str, default=None, metavar="<str>",
        help="set to read (if file exists) or write (if files does not exists) weights",
    )

    parser.add_argument(
        "--rsa_cache_path", type=str, default=None, metavar="<str>",
        help="set to read (if file exists) or write (if files does not exists) RSA values",
    )

    parser.add_argument(
        "--dca_cache_path", type=str, default=None, metavar="<str>",
        help="set to read (if file exists) or write (if files does not exists) DCA parameters and coefficients h and J",
    )

    parser.add_argument(
        "--distance_cache_path", type=str, default=None, metavar="<str>",
        help="set to read (if file exists) or write (if files does not exists) residue-residue distances file (should be a '.npy' file)",
    )

    args = parser.parse_args()


    # Set default output_path
    if args.output_path is None:
        output_dir = "./"
        msa_name:str = os.path.basename(args.msa_path)
        for extension in FastaStream.ACCEPTED_MSA_EXTENSIONS:
            if msa_name.endswith(f".{extension}"):
                msa_name = msa_name.removesuffix(f".{extension}")
                break
        args.output_path = os.path.join(output_dir, f"{msa_name}_structuredca.csv")

    # Verify output_path
    else:
        if not args.output_path.endswith(".csv"):
            raise ValueError(f"ERROR in structuredca: output_path='{args.output_path}' should end with '.csv'.")
        
    # Set None values for 'degenerated' arguments
    if float(args.weights_seqid) == 0.0 or float(args.weights_seqid) == 1.0:
        args.weights_seqid = None
    if float(args.min_seqid) == 0.0:
        args.min_seqid = None


    # Set pLDDT filter
    use_contacts_plddt_filter = args.plddt is not None
    contacts_plddt_cutoff = args.plddt

    # Read mutations if provided
    if args.mfile is not None:
        mutations = read_mutations_file(args.mfile)
    else:
        mutations = None


    # Execute StructureDCA -----------------------------------------------------------
    sdca = StructureDCA(
            args.msa_path,
            args.pdb_path,
            args.chains,
            distance_cutoff=args.distance_cutoff,
            lambda_h=args.lambda_h,
            lambda_J=args.lambda_J,
            exclude_gaps=args.exclude_gaps,
            min_seqid=args.min_seqid,
            weights_seqid=args.weights_seqid,
            use_contacts_plddt_filter=use_contacts_plddt_filter,
            contacts_plddt_cutoff=contacts_plddt_cutoff,
            theta_regularization=args.theta_regularization,
            count_target_sequence=args.count_target_sequence,
            num_threads=args.num_threads,
            distance_cache_path=args.distance_cache_path,
            rsa_cache_path=args.rsa_cache_path,
            weights_cache_path=args.weights_cache_path,
            dca_cache_path=args.dca_cache_path,
            verbose=args.verbose,
            disable_warnings=args.disable_warnings
            )


    # Save scores
    sdca.eval_mutations_table(
            mutations=mutations,
            save_path=args.output_path,
            round_digit=6,
            sep=args.sep,
    )
