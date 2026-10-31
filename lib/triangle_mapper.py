import numpy as np
import os, sys
import argparse

# Add ../lib to sys.path (relative to this script, not CWD)
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "lib"
sys.path.append(str(LIB_DIR))

import tools
from word_db import WordDB

# Acceptable input file extensions
GBK_EXTENSIONS = (".gbk", ".gb", ".gbf", ".gbff")
FASTA_EXTENSIONS = (".fa", ".fsa", ".fas", ".fst", ".fna", ".fasta")

def main(input_folder: str, out_path: str, k: int, min_k: int, max_k: int, matrix_geometry: str = "lower"):
    # Create genome holder
    oWDB = WordDB(min_k=min_k, max_k=max_k)
    # Count k-mers and add to database
    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(GBK_EXTENSIONS) and not file_name.lower().endswith(FASTA_EXTENSIONS):
            continue

        in_path = os.path.join(input_folder, file_name)

        print()
        print(f"File {file_name} is in processes...")

        # Add genome into WordDB
        oWDB.add_genome(in_path)
    
    matrix = oWDB.get_matrix(min_k=k, max_k=k, data_type="digit", digit_map=[-2, -1, 1, 2], matrix_geometry="whole")
    
    coordinates = kmer_triangle_coordinates(matrix=matrix, k=k, matrix_geometry=matrix_geometry)

def kmer_triangle_coordinates(
        matrix: list[list],
        k: int,
        matrix_geometry: str = "lower",
    ) -> list[list]:
    """
    Convert a k-mer abundance matrix into genome coordinates inside
    the k-mer triangle.

    Parameters
    ----------
    matrix : list[list]

        [
            ["Genome", kmer1, kmer2, ...],
            [genome1, v1, v2, ...],
            [genome2, v1, v2, ...],
            ...
        ]

        K-mer titles must end with

            k,x,y

        for example

            TTAA,4,1,1

    k : int
        Word length represented in the matrix.

    matrix_geometry : str
        "lower" or "upper".

    Returns
    -------
    list[list]

        [
            ["Genome","x","y"],
            ["Genome1",x,y],
            ...
        ]
    """

    matrix_geometry = matrix_geometry.lower()

    if matrix_geometry not in ("lower", "upper"):
        raise ValueError(
            "matrix_geometry must be 'lower' or 'upper'."
        )

    n = 2 ** (k + k % 2)

    #
    # Triangle vertices
    #
    v_small = np.array([0.0, 0.0])
    v_large = np.array([1.0, 0.0])
    v_off   = np.array([0.5, np.sqrt(3.0)/2.0])

    centroid = (v_small + v_large + v_off) / 3.0

    #
    # Pre-compute one vector per k-mer
    #
    vectors = []
    used_columns = []

    for column, title in enumerate(matrix[0][1:], start=1):

        parts = str(title).split(",")

        try:
            wl, x, y = map(int, parts[-3:])
        except Exception:
            raise ValueError(
                f"Cannot decode k-mer title '{title}'."
            )

        if wl != k:
            continue

        if matrix_geometry == "lower":

            if x < y:
                continue

            a = (n - x)/(n-1)
            b = (y - 1)/(n-1)
            c = (x - y)/(n-1)

        else:

            if y < x:
                continue

            a = (n - y)/(n-1)
            b = (x - 1)/(n-1)
            c = (y - x)/(n-1)

        pos = (
            a*v_small +
            b*v_large +
            c*v_off
        )

        vectors.append(pos-centroid)
        used_columns.append(column)

    vectors = np.asarray(vectors)

    #
    # Calculate genome coordinates
    #
    result = [["Genome","x","y"]]

    for row in matrix[1:]:

        genome = row[0]

        values = np.asarray(
            [row[c] for c in used_columns],
            dtype=float,
        )

        displacement = values @ vectors

        s = np.sum(np.abs(values))

        if s > 0:
            displacement /= s

        point = centroid + displacement

        result.append([
            genome,
            float(point[0]),
            float(point[1]),
        ])

    return result
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("project", nargs="?", default="example")
    parser.add_argument("-i", "--input", default="input")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-k", "--current_k", type=int, default=4)
    parser.add_argument("--min_k", "-m", type=int, default=3)
    parser.add_argument("--max_k", "-x", type=int, default=7)
    parser.add_argument("--matrix_geometry", "-g", type=str, default="lower")


    args = parser.parse_args()
    
    if not args.min_k <= args.current_k <= args.max_k:
        print(f"\n❌ Error: k {args.current_k} must be within range [{args.min_k}..{args.max_k}]!")
        sys.exit(1)

    # Locate project folder
    project_folder = args.project
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

    main(input_folder=input_path, out_path=outpath, k=args.current_k, min_k=args.min_k, max_k=args.max_k, matrix_geometry=args.matrix_geometry)
