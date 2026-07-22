#!/usr/bin/env python3
import sys
import os
import argparse, shutil

# Add lib directory to path
# Add lib directory to path
path = os.path.abspath(os.path.join(os.getcwd(), "..", "lib"))
sys.path.append(path)
import wdb_sorter_main as main
import tools, wdb_reader, CSV_merger
from dendro import Root as tree

date_of_creation = "September 21, 2025"
__version__ = "1.3"

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
    Select statistically significant k-mers to separate or join taxonomic groups 
    using chi-square testing and FDR correction. Optionally assess group separability 
    via bootstrap classification with RandomForest.

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
        python3 run.py example [options]

    🔧 Options:
        --input,                -i   • Input folder (default: 'input')
        --output                -o   • Output folder (default: 'output')
        --tmp_path              -t   • Temporary folder (default: 'tmp')
        --output_file,          -f   • Output file to save (default: '')
        --output_format,        -r   • Output format newick/pathways (default: 'pathways')
        --max_cluster_number,   -m   • Maximum number of child clusters in internal nodes: (default: 7)
        --max_cluster_content,  -c   • Maximum number of leaves in terminal nodes: (default: 7)
        --max_levels,           -l   • Maximum depth of hierarchy: (default: 5)
        --force_k,              -k   • If set, forces spectral clustering to use exactly 
                                       this number of clusters: (default: 7)

    🆘 Help Options:
        -h, --help            • Show this help message
        -v, --version         • Show version information
    {'='*80}
    """
    print(help_text)

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
    parser.add_argument("project", nargs="?", default="example")
    parser.add_argument("-i", "--input", default="input")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-t", "--tmp_path", default="tmp")
    #parser.add_argument("--output_file", "-f", type=str, default="")
    #parser.add_argument("--output_format", "-r", type=str, default="pathways")
    parser.add_argument("--max_cluster_number", "-m", type=int, default=7)
    parser.add_argument("--max_cluster_content", "-c", type=int, default=7)
    parser.add_argument("--max_levels", "-l", type=int, default=5)
    parser.add_argument("--force_k", "-k", type=int, default=7)


    args = parser.parse_args()

    # Locate project folder
    project_folder = project_name = args.project
    input_path = os.path.join(args.input, project_folder)
    if not os.path.exists(input_path):
        print(f"\n❌ Error: Path '{input_path}' does not exist!")
        sys.exit(1)

    # Locate output folder (reserved for later steps)
    os.makedirs(args.output, exist_ok=True)
    outpath = os.path.join(args.output, project_folder)
    tmp_path = os.path.join(args.tmp_path, project_folder)
    tools.clean(outpath)
    os.makedirs(outpath, exist_ok=True)

    # Locate tmp folder (k-mer matrices will be written here, mirroring leaf structure)
    os.makedirs(args.tmp_path, exist_ok=True)

    '''
    output_file = ""
    output_format = args.output_format.lower()
    if args.output_file and args.cl_algorithm:
        output_file = args.output_file
        extension = ""
        if output_file.rfind(".") > 0:
            extension = output_file[output_file.rfind("."):]
            if output_format == "newick" and extension.upper() != ".NWK":
                extension = ".nwk"
            output_file = output_file[:output_file.rfind(".")]
        output_file = f"{tools.format_file_name(output_file)}.{args.cl_algorithm.lower()}{extension}"
    '''
            
    oKmerSelector = main.kmer_selector(input_path=input_path, 
            out_path=outpath, tmp_path=tmp_path, project=project_folder)
    oKmerSelector.execute(max_cluster_number=args.max_cluster_number, max_cluster_content=args.max_cluster_content, 
        max_levels=args.max_levels, force_k=args.force_k)
