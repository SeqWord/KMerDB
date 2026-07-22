#!/usr/bin/env python3
import sys, os
import argparse, shutil

# Add lib directory to path
path = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
sys.path.append(path)
import main, tools
from dendro import Root as tree

date_of_creation = "September 21, 2025"
__version__ = "1.1"

def show_help():
    help_text = f"""
    {'='*80}
    ■ KMerSelector v{__version__} ■
    ■ Selects most suitable k-mers to separate or join taxonomic groups ■
    {'='*80}
    📅 Created: {date_of_creation}
    👨‍💻 Author: Oleg Reva (oleg.reva@up.ac.za)
       Centre for Bioinformatics and Computational Biology,
       BGM, University of Pretoria, South Africa

    🔬 Purpose:
    Identify long DNA reads by k-mer frequencies.

    ⚙️ Dependencies:
    ┌───────────────────┬───────────────────────┐
    │ Tool              │ Minimum Version       │
    ├───────────────────┼───────────────────────┤
    │ Python            │ 3.12.3                │
    │ sklearn           │ 1.7.0                 │
    │ statsmodels       │ 0.14.5                │
    │ scipy             │ 1.15.2                │
    │ pandas            │ 2.2.3                 │
    │ numpy             │ 2.2.3                 │
    └───────────────────┴───────────────────────┘

    🚀 Usage:
        python3 run.py [options]

    🔧 Options:
        --input,            -i   • Input folder (default: 'input')
        --output,           -o   • Output folder (default: 'output')
        --db_folder,        -f   • Database folder (default: 'db')
        --db_file,          -d   • Database file (default: '')
        --project,          -p   • Optional project name to be used output name.
                                   If empty, input file name will be used (default: '')
        --reads,            -r   • DNA reads in fasta or fastq format (default: '')
        --min_read_length,  -m   • Minimal read length (default: 5000)
        --max_read_length,  -x   • Maximal read length (default: 0)
        --accuracy,         -a   • Border value to select intermediate splits (default: 0.65)
        --specificity,      -s   • Border value for species identification (default: 0.8)
        --use_NN,           -n   • Enable or disable use of neural network (default T/t/Y/y = True, else F/f/N/n = False)
        --entropy_cutoff    -e   • NN prediction entropy should be =< entropy cutoff (default: 0.5)
        
    🆘 Help Options:
        -h, --help            • Show this help message
        -v, --version         • Show version information
    {'='*80}
    """
    print(help_text)
    
def matrix_to_text(matrix: list) -> str:
    """
    Convert matrix = [{rec.id: [species, identity, species_level]}, ...]
    into tab-delimited text with header.

    Output format:
    Record   Taxon   Identity   Path    Species level
    rec.id   
             species identity   path    species_level
    ...
    """
    lines = ["Record\tTaxon\tIdentity\tPath\tSpecies level"]

    for entry in matrix:
        for rec_id, values in entry.items():

            # First line for this record: show rec.id
            lines.append(f"{rec_id}")

            # If there are additional hits for the same record in the same dict:
            # assume values might be a list of lists instead of a single one
            if values and isinstance(values[0], list):
                values.sort(key=lambda ls: float(ls[1]), reverse=True)
                for sub in values:
                    if len(sub) != 4:
                        continue
                    sp, ident, path, level = sub
                    lines.append(f"\t{sp}\t{ident}\t{path}\t{level}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Check for help/version manually before argparse
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            show_help()
            sys.exit()
        if sys.argv[1] in ("-v", "--version"):
            print(f"🌟 VARCALL v{__version__} | Created: {date_of_creation}")
            sys.exit()

    # Manual parser setup (disabling default help/version)
    parser = argparse.ArgumentParser(add_help=False)
    
    # 🔧 Options (exactly those in the help)
    parser.add_argument("-i", "--input",   default="input", help="Input folder (default: 'input')")
    parser.add_argument("-o", "--output",  default="output", help="Output folder (default: 'output')")
    parser.add_argument("-f", "--db_folder", default="db",  help="Database folder (default: 'db')")
    parser.add_argument("-d", "--db_file", required=True,   help="Database file (required)")
    parser.add_argument("-r", "--reads",   required=True,   help="DNA reads in fasta/fastq (required)")
    parser.add_argument("-p", "--project", default="", help="Optional project name (default: '')")
    parser.add_argument("-m", "--min_read_length", type=int, default=5000, help="Minimal read length (default: 5000)")
    parser.add_argument("-x", "--max_read_length", type=int, default=0,    help="Maximal read length (default: 0)")
    parser.add_argument("-a", "--accuracy",   type=float, default=0.5, help="Border value to select intermediate splits (default: 0.65)")
    parser.add_argument("-s", "--specificity", type=float, default=0.7,  help="Border value for species identification (default: 0.8)")
    parser.add_argument("-n", "--use_NN", type=tools.str2bool, choices=["T", "t", "F", "f", "Y", "y", "N", "n"], default=True)
    parser.add_argument("-e", "--entropy_cutoff", type=float, default=0.5,  help="Top cutoff entropy value for accepting NN classification (default: 0.5)")
    
    args = parser.parse_args()
    
    # ---- Post-parse validation ----
    def _fail(msg: str) -> None:
        print(f"❌ {msg}\n")
        show_help()
        sys.exit(2)
    
    # Ensure output folder exists
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        _fail(f"Cannot create output folder '{args.output}': {e}")
    
    # Validate read length settings
    if args.min_read_length is None or int(args.min_read_length) <= 0:
        _fail("--min_read_length must be a positive integer")
    if args.max_read_length is None or int(args.max_read_length) < 0:
        _fail("--max_read_length must be >= 0")
    if int(args.max_read_length) != 0 and int(args.max_read_length) <= int(args.min_read_length):
        _fail("--max_read_length must be greater than --min_read_length (or 0 to disable the cap)")
    if float(args.entropy_cutoff) < 0 or float(args.entropy_cutoff) > 1:
        _fail("Entropy cutoff value must be in range [0 : 1]")
    
    # Validate accuracy & specificity in [0,1]
    def _in_unit_interval(name: str, val: float) -> None:
        if val < 0.0 or val > 1.0:
            _fail(f"--{name} must be in the range [0, 1], got {val}")
    
    _in_unit_interval("accuracy", args.accuracy)
    _in_unit_interval("specificity", args.specificity)
    
    # Validate input/db paths exist
    reads_path = os.path.join(args.input, args.reads)
    db_path    = os.path.join(args.db_folder, args.db_file)
    
    if not os.path.isdir(args.input):
        _fail(f"Input folder not found: {args.input}")
    if not os.path.isfile(reads_path):
        _fail(f"--reads file not found at: {reads_path}")
    if not os.path.isdir(args.db_folder):
        _fail(f"Database folder not found: {args.db_folder}")
    if not os.path.isfile(db_path):
        _fail(f"--db_file not found at: {db_path}")
    
    #### EXECUTE READ IDENTIFICATION
    project_name = args.project 
    if not project_name:
        project_name = os.path.basename(reads_path)
        if project_name.rfind(".") > 1:
            project_name = project_name[:project_name.rfind(".")]
    fname, oDB, supplementary = tools.openDBFile(db_path)
    
    # matrix = [{rec.id : [species, identity float, species_level True/False]}, ...]
    matrix, log = oDB.identify(open(reads_path), 
        min_length=int(args.min_read_length), max_length=int(args.max_read_length),
        specificity=float(args.specificity), accuracy=float(args.accuracy),
        flg_use_NN=args.use_NN, entropy_cutoff=float(args.entropy_cutoff)
    )
    
    # Save output
    output_file = os.path.join(args.output, f"{project_name}.txt") 
    output_log_file = os.path.join(args.output, f"{project_name}.log.txt") 
    tools.saveTextFile(matrix_to_text(matrix), output_file)
    tools.saveTextFile("\n".join(log), output_log_file)
    
