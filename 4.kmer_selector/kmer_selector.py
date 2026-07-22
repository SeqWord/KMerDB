#!/usr/bin/env python3
import sys
import os
import argparse, shutil
from pathlib import Path  

# Add ../lib to sys.path (relative to this script file, not CWD)
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "lib"
sys.path.append(str(LIB_DIR))
import main, tools

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
        --input,         -i   • Input folder (default: 'input')
        --output         -o   • Output folder (default: 'output')
        --tmp_path       -t   • Temporary folder (default: 'tmp')
        --diverse_words, -d   • Select diverse words (T/F), default: T
        --common_words,  -c   • Select diverse words (T/F), default: T
        --min_k,         -x   • Minimal k-value, default 4
        --max_k,         -y   • Maximal k-value, default 8
        --top_selected,  -t   • Maximal number of selected k-mers, default: 100
        --min_selected,  -l   • Manimal number of selected k-mers, default: 10
        --use_NN,        -n   • Enable or disable use of neural network (default T/t/Y/y = True, else F/f/N/n = False)
        --mmcs,          -m   • Minimal median cluster size to apply NN model, default 30
        --diverse_border,     • Cutoff border of diverse k-mers, default: 0.5 -> [-0.5..0.5]
        --common_border,      • Cutoff border of common k-mers, default: 0.8 -> [-1..-0.8] + [0.8..1]
        --level_increment,    • Incrment of common and diverse k-mer borders with each level: 0.2
        --step_by_step,       • Step by step execution (T/F), default: F

    ■ Reserved:
        --permutations,       • Minimal number of differences between k-mers, default 1
        --shift,              • Minimal number of sliding between k-mers, default 1
        --constituents,       • Minimal number of difference between constituent k-mers, default 1
        --p_value,       -p   • Filter by maximum p-value (e.g. 0.05), default None
        --significance,  -s   • Show only FDR-significant results (TRUE/FALSE), default None

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
    parser.add_argument("--tmp_path", default="tmp")
    parser.add_argument("--min_k", "-x", type=int, default=4)
    parser.add_argument("--max_k", "-y", type=int, default=8)
    parser.add_argument("-t", "--top_selected", type=str, default=100)
    parser.add_argument("-l", "--min_selected", type=str, default=10)
    parser.add_argument("-n", "--use_NN", type=tools.str2bool, choices=["T", "t", "F", "f", "Y", "y", "N", "n"], default=True)
    parser.add_argument("-m", "--mmcs", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--constituents", type=int, default=1)
    parser.add_argument("--p_value", "-p", type=float, default=None)
    parser.add_argument("--significance", "-s", type=lambda x: x.lower() in ('true', 't', '1'), default=None)
    parser.add_argument("--diverse_border", type=float, default=.5)
    parser.add_argument("--common_border", type=float, default=.8)
    parser.add_argument("--level_increment", type=float, default=0.2)
    parser.add_argument("--step_by_step", "-b", type=lambda x: x.lower() in ('true', 't', '1'), default=False)

    args = parser.parse_args()

    # Check argument settings
    min_k, max_k = [tools.ascertain_integer(v, 1) for v in [args.min_k, args.max_k]]
    if not all([min_k, max_k]):
        print(f"\n❌ Error: min_k and max_k '{min_k}, {max_k}' must be integers > 1!")
        sys.exit(1)
    if min_k >= max_k:
        print(f"\n❌ Error: min_k must be smaller than max_k: '{min_k}, {max_k}' !")
        sys.exit(1)
        
    try:
        top_selected = int(args.top_selected)
        min_selected = int(args.min_selected)
    except:
        print(f"Check, either --min_selected {args.min_selected} or --top_selected {args.top_selected} are not integers!")
        sys.exit(1)
        
    if min_selected > top_selected:
        print(f"Check, --min_selected {args.min_selected} must be <= --top_selected {args.top_selected}!")
        sys.exit(1)
        
    try:
        mmcs = abs(int(args.mmcs))
    except:
        print(f"Minimal median cluster size (mmcs = {args.mmcs}) must be a positive integer!")
        sys.exit(1)

    try:    
        diverse_border = abs(float(args.diverse_border))
        common_border = abs(float(args.common_border))
    except:
        print(f"Diverse and common borders must be in range from 0 to 1. Current values are: [{args.diverse_border}, {args.common_border}]")
        sys.exit(1)
    if not all([0 <= diverse_border <= 1, 0 <= common_border <= 1]):
        print(f"Diverse and common borders must be in range from 0 to 1. Current values are: [{args.diverse_border}, {args.common_border}]")
        sys.exit(1)
        
    diverse_top = diverse_border
    diverse_bottom = -diverse_border
    common_top = common_border
    common_bottom = -common_border

    p_value = tools.ascertain_float(args.p_value) if args.p_value is not False else False

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

    # Merge and filter k-mers
    #### TODO: filter setting should be set by arguments
    filter_settings = {
            'top' : 0,                                  # Size limit, absolute or as percentage
            'permutations' : int(args.permutations),    # Number of different nucleotides in a pair of words
            'frameshift' : int(args.shift),             # Filter shifted words
            'constituents' : int(args.constituents),    # Filter constituent sub-words
            'wl_min' : min_k,                           # Minimal k-mer length
            'wl_max' : max_k,                           # Maximal k-mer length
            'redundancy' : 0,                           # Word redundancy
            'similarity' : 0,                           # Maximal similarity between words
            'threshold' : 0,                            # Value threshold
        }
        
    oKmerSelector = main.kmer_selector(input_path=input_path, 
            out_path=outpath, tmp_path=tmp_path, project=project_folder, 
            min_k=min_k, max_k=max_k, flg_use_NN=args.use_NN, mmcs=mmcs,
            min_selected = min_selected, top_selected = top_selected, 
            diverse_bottom=diverse_bottom, diverse_top=diverse_top,
            common_bottom=common_bottom, common_top=common_top, level_increment=args.level_increment,
            filter_settings=filter_settings)
    oKmerSelector.execute(flg_print_classifier = True, flg_step_by_step = args.step_by_step)
