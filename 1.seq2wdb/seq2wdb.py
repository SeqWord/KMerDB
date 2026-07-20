#!/usr/bin/env python3
from __future__ import annotations
import os, sys, shutil
import argparse
from pathlib import Path
from typing import Tuple, List
from datetime import datetime

# Add ../lib to sys.path (relative to this script, not CWD)
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "lib"
sys.path.append(str(LIB_DIR))

import tools
from word_db import WordDB as wdb  # type: ignore

# Acceptable input file extensions
GBK_EXTENSIONS = (".gbk", ".gb", ".gbf", ".gbff")
FASTA_EXTENSIONS = (".fa", ".fsa", ".fas", ".fst", ".fna", ".fasta")

def get_lineage(path: str) -> str:
    """
    Extract lineage from the FIRST GenBank record in `path` and return it as a string
    joined with '|'. Tries, in order:
      1) record.annotations['taxonomy'] (list of ranks)
      2) 'source' feature's 'organism'
      3) record.annotations['organism']
    Always appends '|<species>|<accession>' to the lineage string.
    """
    try:
        from Bio import SeqIO
    except Exception:
        return ""
        
    extension = path[str(path).rfind("."):].lower()
        
    try:
        with open(path, "rt", encoding="utf-8", newline=None) as fh:
            if extension in (".fa",".fsa",".fas",".fst",".fna",".fasta"):
                rec = next(SeqIO.parse(fh, "fasta"), None)
                return rec.description
            if extension in (".gbk", ".gbf", ".gbff", ".gb"):
                rec = next(SeqIO.parse(fh, "genbank"), None)
            else:
                return ""
    except Exception:
        return ""

    if rec is None:
        return ""

    # Species name
    species = rec.annotations.get("organism", "")
    if not species and rec.features:
        for feat in rec.features:
            if feat.type == "source" and "organism" in feat.qualifiers:
                species = feat.qualifiers["organism"][0]
                break
    species = str(species).strip()

    # Accession number
    accession = ""
    try:
        accession = rec.annotations.get("accessions", [""])[0]
    except Exception:
        pass
    if not accession and hasattr(rec, "id"):
        accession = rec.id
    accession = str(accession).strip()

    # 1) Try taxonomy
    lineage = ""
    tax = rec.annotations.get("taxonomy")
    if isinstance(tax, (list, tuple)) and tax:
        lineage = "|".join(str(x).strip() for x in tax if str(x).strip())
    else:
        # 2) Try species from features
        if species:
            lineage = species
        else:
            # 3) Fallback to annotations['organism'] (already handled as species)
            lineage = species

    # Append species + accession if available
    extras = []
    if species:
        extras.append(species)
    if accession:
        extras.append(accession)

    if extras:
        lineage = f"{lineage}|{'|'.join(extras)}" if lineage else "|".join(extras)

    return lineage

def process(project_folder: str, 
        input_folder: str, 
        output_folder: str, 
        min_k: int, 
        max_k: int, 
        targer_seq_length: int, 
        chunk_number: int, 
        matrix_geometry: str, 
        pack_files: str, 
        flg_concatenate: bool = True) -> None:
    """
    Build a WordDB from GenBank files in `input_folder` and save to `output_folder`.
    """
    outpath = ""
    if pack_files == 'single':
        outpath = os.path.join(output_folder, f"{project_folder}.wdb")
        # Prepare database
        oWD = wdb(title="", date=datetime.now().strftime("%d-%m-%Y"), matrix_geometry=matrix_geometry)
    elif pack_files == 'multiple':
        os.makedirs(output_folder, exist_ok=True)
        output_folder = os.path.join(output_folder, project_folder)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)   # deletes the whole folder and contents
        os.makedirs(output_folder)
            
    for file_name in sorted(os.listdir(input_folder)):
        if not file_name.lower().endswith(GBK_EXTENSIONS) and not file_name.lower().endswith(FASTA_EXTENSIONS):
            continue

        in_path = os.path.join(input_folder, file_name)

        lineage = get_lineage(in_path)
        print()
        print(f"File {file_name} is in processes...")

        # Read sequence file
        genome_collection = tools.openSeqFile(in_path, flg_concatenate)
        if not genome_collection:
            return

        for accession in genome_collection:
            print(f"\t{accession}...")
            dataset = {'Sequence' : genome_collection[accession].upper()}
            if pack_files == 'multiple':
                base_name = tools.format_file_name(accession)
                outpath = os.path.join(output_folder, f"{base_name}.wdb")
                # Prepare database
                oWD = wdb(title=base_name, date=datetime.now().strftime("%d-%m-%Y"), matrix_geometry=matrix_geometry)
                # Add genome into WordDB
                oWD.add_genome(path=in_path, 
                    dataset=dataset, accession=accession, 
                    lineage=lineage, 
                    wl1=min_k, 
                    wl2=max_k, 
                    targer_seq_length=targer_seq_length, 
                    chunk_number=chunk_number)
                # Save DB
                oWD.save_dbfile(outpath)
            elif pack_files == 'single':
                # Add genome into WordDB
                oWD.add_genome(in_path, 
                    dataset=dataset, accession=accession, 
                    lineage=lineage, 
                    wl1=min_k, 
                    wl2=max_k, 
                    targer_seq_length=targer_seq_length, 
                    chunk_number=chunk_number)
            print()

    if pack_files == 'single':
        # Save DB
        oWD.title = "|".join(Path(input_folder).parts[-2:])  # e.g., "<parent>|<leaf>"
        oWD.save_dbfile(outpath)
        print(f"✅ Saved: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a word DB (k-mer DB) from GenBank files in a project folder."
    )
    # 1) positional
    parser.add_argument(
        "project_folder",
        help="Project subfolder inside input_folder containing .gbk/.gb/.gbf files",
    )
    # 2) optionals with defaults
    parser.add_argument(
        "--input_folder", "-i", default="input", help="Base input folder (default: input)"
    )
    parser.add_argument(
        "--output_folder", "-o", default="output", help="Base output folder (default: output)"
    )
    parser.add_argument(
        "--min_k", "-n", type=int, default=4, help="Minimal k-mer size (default: 4)"
    )
    parser.add_argument(
        "--max_k", "-x", type=int, default=7, help="Maximal k-mer size (default: 7)"
    )
    parser.add_argument(
        "--chunk_number", "-c", type=int, default=20, help="Number of genomic fragment (chunks) (default: 20)"
    )
    parser.add_argument(
        "--target_seq_length", "-l", type=str, default="500k", help="Target sequence length (default: '500k', if 0 - entire sequence)"
    )
    parser.add_argument(
        "--do-not-concatenate",
        dest="concatenate",
        action="store_false",
        help="Do NOT concatenate sequences of multiple records"
    )
    parser.set_defaults(concatenate=True)
    
    """
    The assumption is that k-mers and reverse complement k-mer are equally frequent in bacterial/archaeal chromosomes.
    Setting options 'lower' or 'upper' remove all reverse complement k-mer duplicates from consideration.
    This assumprion may not be true for plasmids and viruses. Use 'whole' instead.
    """
    parser.add_argument(
        "--matrix_geometry", "-m", type=str, default="lower", choices=['lower', 'upper', 'whole'], 
        help="'lover' | 'upper' | 'whole' (default: 'lower')"
    )
    parser.add_argument(
        "--pack_files", "-f", type=str, default='single', choices=["single", "multiple"], 
            help="Pack k-mer patterns into a single or multiple files (default: 'single')"
    )

    args = parser.parse_args()

    # Resolve folders
    project_folder = args.project_folder
    input_dir = os.path.join(args.input_folder, project_folder)
    output_dir = args.output_folder

    # Basic checks
    if args.min_k <= 0 or args.max_k <= 0 or args.max_k < args.min_k:
        parser.error("--min_k and --max_k must be positive and max_k >= min_k")

    if not os.path.isdir(input_dir):
        parser.error(f"Input project folder not found: {input_dir}")
    
    try:
        chunk_number = int(args.chunk_number)
    except:
        parser.error(f"Chunk number {args.chunk_number} must be an integer!")
        
    try:
        min_k = int(args.min_k)
        max_k = int(args.max_k)
    except:
        parser.error(f"Check min_k {args.min_k} and max_k {args.max_k} must be integers!")
        
    target_seq_length = str(args.target_seq_length)
    unit = target_seq_length[-1]
    if unit.lower() in ('k', 'm'):
        target_seq_length = target_seq_length[:-1]
    try:
        target_seq_length = int(target_seq_length)
    except:
        parser.error(f"Target sequence length {target_seq_length} must be in format '10000', '100k' or '100m'!")
    if unit == 'k':
        target_seq_length *= 10**3
    if unit == 'm':
        target_seq_length *= 10**6

    process(project_folder, input_dir, output_dir, min_k, max_k, target_seq_length, chunk_number, args.matrix_geometry, args.pack_files, args.concatenate)

if __name__ == "__main__":
    main()
