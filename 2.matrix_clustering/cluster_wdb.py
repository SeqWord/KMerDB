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
    parser.add_argument("project", nargs="?", default="example")
    parser.add_argument("-i", "--input", default="input")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("--min_k", "-x", type=int, default=4)
    parser.add_argument("--max_k", "-y", type=int, default=8)
    parser.add_argument("--output_file", "-f", type=str, default="")
    parser.add_argument("--output_format", "-r", choices=["newick", "pathways", "Newick", "Pathways"], default="newick")
    parser.add_argument("--cl_algorithm", "-a", choices=["UPGMA", "NJ", "SPECTRAL", "upgma", "nj", "spectral"], default="spectral")

    args = parser.parse_args()

    input_path = os.path.join(args.input, args.project)
    if not os.path.exists(input_path):
        print(f"Input folder {input_path} does not exist!")
        sys.exit(1)

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # Use --output_file if given; else default to <output>/<project>
    extension = ".txt"
    if args.output_format.lower() == "newick":
        extension = ".nwk"
    output_base = args.output_file if args.output_file else os.path.join(out_path, f"{args.project}{extension}")

    # Execute tree building
    make_tree.execute(
        input_path,
        min_k=args.min_k,
        max_k=args.max_k,
        algorithm=args.cl_algorithm,           # choices already enforce case
        output_file=output_base,               # ✅ use chosen base
        output_format=args.output_format       # already one of the choices
    )

    print(f"✅ Tree file was saved to {output_base}")
