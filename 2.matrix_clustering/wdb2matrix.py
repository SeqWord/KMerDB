#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path  

# Add ../lib to sys.path (relative to this script file, not CWD)
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "lib"
sys.path.append(str(LIB_DIR))

import make_tree  # assumes make_tree.py is in ../lib

if __name__ == "__main__":
    # Manual parser setup (disabling default help/version)
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument(
        "project", nargs="?", default="example",
        help="Project folder name (default: example). Interpreted as ./input/<project> in your code."
    )
    parser.add_argument("-i", "--input", default="input", help="INPUT folder (default: ./input)")
    parser.add_argument("-o", "--output", default="output", help="OUTPUT folder (default: ./output)")
    parser.add_argument("--min_k", "-x", type=int, default=4, help="Minimal k (included), default: 4")
    parser.add_argument("--max_k", "-y", type=int, default=5, help="Maximal k (excluded), default: 5")
    
    parser.add_argument(
        "--output_file", "-f", type=str, default="",
        help="Output matrix file name (default: '' -> project folder name is used)"
    )
    parser.add_argument(
        "--output_format", "-t", choices=["csv", "CSV", "tsv", "TSV"], default="tsv",
        help="Matrix format: TSV or CSV (default: tsv)"
    )
    
    args = parser.parse_args()

    input_path = os.path.join(args.input, args.project)
    if not os.path.exists(input_path):
        print(f"Input folder {input_path} does not exist!")
        sys.exit(1)

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # Use --output_file if given; else default to <output>/<project>
    extension = ".tsv"
    if args.output_format.lower() == "csv":
        extension = ".csv"
    
    output_file = args.output_file if args.output_file else f"{args.project}{extension}"
    if not output_file.endswith(extension):
        output_file += extension
    output_file = os.path.join(out_path, output_file)

    # Execute tree building
    make_tree.execute(
        input_path,
        min_k=args.min_k,
        max_k=args.max_k,
        algorithm="",                            # NA
        output_file=output_file,                 # ✅ use chosen base
        output_format=args.output_format.lower(),# already one of the choices
        matrix_type="V"                          # request to create value matrix
    )

    print(f"✅ Matrix was saved to {output_base}")
