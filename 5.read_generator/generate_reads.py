#!/usr/bin/env python3
"""
PseudoRead Generator
Generate artificial DNA reads from GenBank genomes (.gbk/.gb/.gbf) as FASTA or FASTQ.
"""

import os
import sys
import argparse
import random
from typing import List, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ──────────────────────────────────────────────────────────────────────────────
# Metadata & Help/Version
# ──────────────────────────────────────────────────────────────────────────────

date_of_creation = "September 21, 2025"
__version__ = "1.0"


def show_help():
    help_text = f"""
    {'='*80}
    ■ PseudoRead Generator v{__version__} ■
    ■ Generate artificial DNA reads from GenBank genomes ■
    {'='*80}
    📅 Created: {date_of_creation}
    👨‍💻 Author: Oleg Reva (oleg.reva@up.ac.za)
       Centre for Bioinformatics and Computational Biology,
       BGM, University of Pretoria, South Africa

    🔬 Purpose:
    This tool generates pseudo-reads from annotated genomes stored in
    GenBank format (.gbk, .gb, .gbf). Reads are randomly sampled with
    configurable lengths, numbers, and species diversity. Circularity of
    the genomes is assumed. FASTA and FASTQ formats are supported.

    🚀 Usage:
        python3 generate_reads.py project_folder output_format (fasta|fastq) [options]

    🔧 Options:

        Positional arguments:
          project_folder           Subfolder inside 'input' that contains
                                   GenBank files (.gbk, .gb, .gbf).

        Optional:
          -n, --read_number        Total number of pseudo-reads to generate
                                   (default: 10000)
          -f, --output_format      Output format: 'fasta' or 'fastq'.
          --min_length             Minimal read length (default: 5000)
          --max_length             Maximal read length (default: 20000)
          --species_number         Number of species/files to sample. If 0,
                                   all available files are used (default: 0)
          --input_folder           Base input folder (default: 'input')
          --output_folder          Base output folder (default: 'output')
          --seed                   Random seed for reproducibility (default: none)

    📝 Notes:
      • If requested species_number exceeds available files, it will be
        clamped to the maximum.
      • Reads are generated assuming genome circularity.
      • FASTQ qualities are randomly sampled from Q20–Q40.
      • Output files are saved as 'output/<project>.fasta' or 'output/<project>.fastq'.

    🆘 Help Options:
        -h, --help                 Show this help message
        -v, --version              Show version information
    {'='*80}
    """
    print(help_text)


def show_version():
    print(f"🌟 PseudoRead Generator v{__version__} | Created: {date_of_creation}")


# ──────────────────────────────────────────────────────────────────────────────
# Core functionality
# ──────────────────────────────────────────────────────────────────────────────

def generate_pseudoreads(
    project_folder: str,
    output_format: str,
    read_number: int = 10000,
    min_length: int = 5000,
    max_length: int = 20000,
    species_number: int = 0,
    input_folder: str = 'input',
    output_folder: str = 'output',
) -> None:
    """
    Generate pseudo-reads from circular genomes annotated in GenBank files.

    Parameters
    ----------
    project_folder : str
        Name of subfolder in `input_folder` that contains GenBank files (.gbk, .gb, .gbf).
    output_format : str
        'fasta' or 'fastq' (case-insensitive).
    read_number : int
        Total number of pseudo-reads to generate across all selected species/files.
    min_length, max_length : int
        Minimum and maximum read lengths (inclusive).
    species_number : int
        If 0, use all available input files; otherwise randomly choose this many files.
        If larger than available, it will be clamped to the number of available files.
    input_folder : str
        Base input directory.
    output_folder : str
        Base output directory.

    Notes
    -----
    - Reads are sampled assuming circular sequences (wrap-around if needed).
    - FASTQ qualities are sampled uniformly from Phred Q20..Q40 and emitted as Sanger (ASCII+33).
    - Header format:
        FASTA:  >{species_name} {basename.gbk} [{start}..{stop}]
        FASTQ:  @{species_name} {basename.gbk} [{start}..{stop}]
    """
    # Validate arguments
    fmt = output_format.lower()
    if fmt not in ("fasta", "fastq"):
        raise ValueError("output_format must be 'fasta' or 'fastq'")

    if read_number <= 0:
        raise ValueError("read_number must be positive")
    if min_length <= 0 or max_length <= 0 or max_length < min_length:
        raise ValueError("min_length/max_length must be positive and max_length >= min_length")

    in_dir = os.path.join(input_folder, project_folder)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input project folder not found: {in_dir}")

    os.makedirs(output_folder, exist_ok=True)

    # 1) list suitable files
    exts = {".gbk", ".gb", ".gbf"}
    files: List[str] = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if os.path.isfile(os.path.join(in_dir, f)) and os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        raise FileNotFoundError(f"No GenBank files (.gbk/.gb/.gbf) found in: {in_dir}")

    # 2) reduce to desired species_number
    available = len(files)
    if species_number <= 0 or species_number > available:
        species_number = available
    else:
        files = random.sample(files, species_number)

    # 3) assign counts per file with cap ~ 2*read_number / n_files
    n_files = len(files)
    cap = max(1, (2 * read_number) // n_files)
    counts = [1] * n_files
    remaining = read_number - n_files
    while remaining > 0:
        i = random.randrange(n_files)
        if counts[i] < cap:
            counts[i] += 1
            remaining -= 1
        if all(c == cap for c in counts) and remaining > 0:
            # distribute leftovers ignoring cap
            for _ in range(remaining):
                counts[random.randrange(n_files)] += 1
            remaining = 0

    # Helpers

    def pick_species_name(rec: SeqRecord, fallback_file: str) -> str:
        # Try annotations['organism']
        org = rec.annotations.get("organism")
        if org:
            return str(org)
        # Try source feature 'organism'
        for feat in rec.features:
            if feat.type == "source":
                org = feat.qualifiers.get("organism")
                if org:
                    return str(org[0])
        # Fallback to file stem
        return os.path.splitext(os.path.basename(fallback_file))[0]

    def circular_slice(seq: Seq, start: int, length: int) -> Seq:
        n = len(seq)
        if length <= n - start:
            return seq[start:start + length]
        # wrap
        end = (start + length) % n
        return seq[start:] + seq[:end]

    def choose_record_weighted(records: List[SeqRecord]) -> SeqRecord:
        # Weighted by sequence length (prefer longer records within a file)
        weights = [max(1, len(r.seq)) for r in records]
        total = sum(weights)
        r = random.uniform(0, total)
        cum = 0.0
        for rec, w in zip(records, weights):
            cum += w
            if r <= cum:
                return rec
        return records[-1]

    def random_quality(length: int) -> str:
        # Uniform Q20..Q40 → ASCII (chr(q + 33))
        return "".join(chr(random.randint(20, 40) + 33) for _ in range(length))

    # 4–6) sample reads file-by-file, then shuffle
    all_reads_fasta: List[Tuple[str, str]] = []           # (header, sequence)
    all_reads_fastq: List[Tuple[str, str, str]] = []      # (header, sequence, quality)

    for fpath, n_reads in zip(files, counts):
        # Load all records in this file
        records = list(SeqIO.parse(fpath, "genbank"))
        # Filter out empty sequences
        records = [r for r in records if len(r.seq) > 0]
        if not records:
            continue

        basename = os.path.basename(fpath)

        for _ in range(n_reads):
            rec = choose_record_weighted(records)
            Lmin = min_length
            Lmax = min(max_length, len(rec.seq))  # cap not to exceed record length
            if Lmin > Lmax:
                # If requested min_length exceeds record length, clamp to record length
                Lmin = 1
                Lmax = len(rec.seq)
            L = random.randint(Lmin, Lmax)
            start = random.randint(0, len(rec.seq) - 1)
            seq = str(circular_slice(rec.seq, start, L))
            stop = (start + L) % len(rec.seq)

            species_name = pick_species_name(rec, fpath)
            header_core = f"{species_name} {basename} [{start}..{stop}]"

            if fmt == "fasta":
                all_reads_fasta.append((f">{header_core}", seq))
            else:
                qual = random_quality(L)
                all_reads_fastq.append((f"@{header_core}", seq, qual))

    # Shuffle across all files
    random.shuffle(all_reads_fasta)
    random.shuffle(all_reads_fastq)

    # 7) write output
    project_name = os.path.basename(os.path.normpath(project_folder))
    if fmt == "fasta":
        out_path = os.path.join(output_folder, f"{project_name}.fasta")
        with open(out_path, "w", encoding="utf-8") as out:
            for h, s in all_reads_fasta:
                out.write(f"{h}\n{s}\n")
    else:
        out_path = os.path.join(output_folder, f"{project_name}.fastq")
        with open(out_path, "w", encoding="utf-8") as out:
            for h, s, q in all_reads_fastq:
                out.write(f"{h}\n{s}\n+\n{q}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Handle help/version explicitly (before argparse), per your preferred style
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            show_help()
            sys.exit(0)
        if sys.argv[1] in ("-v", "--version"):
            show_version()
            sys.exit(0)

    parser = argparse.ArgumentParser(add_help=False)
    # Required (as positionals to match "Required" in help)
    parser.add_argument("project_folder", help="Subfolder inside 'input' containing GenBank files")

    # Optional
    parser.add_argument("-n", "--read_number", type=int, default=10000,
                        help="Total number of pseudo-reads (default: 10000)")
    parser.add_argument("-f", "--output_format", type=str, default="fasta", choices=["fasta", "fastq"], 
                        help="Output format fasta | fastq (default: 'fasta')")
    parser.add_argument("--min_length", type=int, default=5000,
                        help="Minimal read length (default: 5000)")
    parser.add_argument("--max_length", type=int, default=20000,
                        help="Maximal read length (default: 20000)")
    parser.add_argument("--species_number", type=int, default=0,
                        help="Number of species/files to use (0 = all)")
    parser.add_argument("--input_folder", default="input",
                        help="Base input folder (default: 'input')")
    parser.add_argument("--output_folder", default="output",
                        help="Base output folder (default: 'output')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: none)")

    args = parser.parse_args()

    # Basic validations & environment prep
    if args.seed is not None:
        random.seed(args.seed)

    # Folders
    in_dir = os.path.join(args.input_folder, args.project_folder)
    if not os.path.isdir(in_dir):
        print(f"❌ Input project folder not found: {in_dir}\n")
        show_help()
        sys.exit(2)

    try:
        os.makedirs(args.output_folder, exist_ok=True)
    except Exception as e:
        print(f"❌ Cannot create output folder '{args.output_folder}': {e}\n")
        show_help()
        sys.exit(2)

    # Ranges
    if args.read_number <= 0:
        print("❌ --read_number must be positive\n")
        show_help()
        sys.exit(2)
    if args.min_length <= 0 or args.max_length <= 0 or args.max_length < args.min_length:
        print("❌ --min_length and --max_length must be positive and max_length >= min_length\n")
        show_help()
        sys.exit(2)

    # Execute
    generate_pseudoreads(
        project_folder=args.project_folder,
        output_format=args.output_format,
        read_number=args.read_number,
        min_length=args.min_length,
        max_length=args.max_length,
        species_number=args.species_number,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
    )
