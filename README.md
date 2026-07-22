# KMerDB
Machine-learning pipeline for taxonomic classification and binning of DNA fragments using k-mer abundance patterns

# K-mer Based Taxonomic Identification Pipeline

A machine-learning pipeline for taxonomic identification of DNA sequences using genome-specific k-mer frequency patterns.

The pipeline converts reference genomes into k-mer frequency databases, clusters related genomes, identifies informative k-mers, trains hierarchical classifiers, generates synthetic sequencing reads for validation, and classifies unknown reads.

---

## Pipeline Overview

```
Reference genomes (.gbk/.gb/.gbf)
              │
              ▼
1. Build WDB (k-mer database)
              │
              ▼
2. Cluster genomes
              │
              ▼
3. Sort WDB files into hierarchical groups
              │
              ▼
4. Select informative k-mers and train classifiers
              │
              ▼
5. Generate pseudo-reads
              │
              ▼
6. Classify unknown reads
```

---

# Step 1 — Build k-mer Word Database

**Program**

```
1.seq2wdb/seq2wdb.py
```

**Example**

```bash
cd 1.seq2wdb
python seq2wdb.py project_folder [options]
```

### Purpose

Counts k-mer frequencies from GenBank files and stores them in a Word Database (`.wdb`), which serves as the foundation for downstream analyses.

### Required argument

| Argument | Description |
|----------|-------------|
| `project_folder` | Subfolder inside `input/` containing `.gbk`, `.gb`, or `.gbf` files |

### Main optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i`, `--input_folder` | `input` | Input directory |
| `-o`, `--output_folder` | `output` | Output directory |
| `--min_k` | 4 | Minimum k-mer size |
| `--max_k` | 6 | Maximum k-mer size |
| `-c`, `--chunk_number` | 20 | Number of genome fragments used to create a probe sequence |
| `-l`, `--target_seq_length` | 500k | Target probe sequence length |
| `--do-not-concatenate` | — | Use only the first sequence record |
| `-m`, `--matrix_geometry` | lower | lower / upper / whole |
| `-f`, `--pack_files` | single | Save one combined WDB or separate WDB files |

---

# Step 2 — Cluster k-mer Patterns

**Program**

```
2.create_tree/create_tree.py
```

**Example**

```bash
cd 2.create_tree
python create_tree.py project [options]
```

### Purpose

Clusters genomes using UPGMA, Neighbor Joining (NJ), or Spectral Clustering and generates either a Newick tree or cluster assignments.

### Main arguments

| Argument | Default |
|----------|---------|
| `project` | example |
| `-i`, `--input` | input |
| `-o`, `--output` | output |
| `--min_k`, `--max_k` | 4–8 |
| `-f`, `--output_file` | automatic |
| `-r`, `--output_format` | newick |
| `-a`, `--cl_algorithm` | spectral |

---

# Step 3 — Sort WDB Files

**Program**

```
3.WDB_sorter/sorter.py
```

**Example**

```bash
cd 3.WDB_sorter
python sorter.py project [options]
```

### Purpose

Uses spectral clustering to recursively organize WDB files into hierarchical subfolders according to k-mer similarity.

### Main arguments

| Argument | Default |
|----------|---------|
| `project` | example |
| `-i`, `--input` | input |
| `-o`, `--output` | output |
| `-t`, `--tmp_path` | tmp |
| `-m`, `--max_cluster_number` | 7 |
| `-c`, `--max_cluster_content` | 7 |
| `-l`, `--max_levels` | 5 |
| `-k`, `--force_k` | 7 |

---

# Step 4 — Select Informative k-mers

**Program**

```
4.kmer_selector/kmer_selector.py
```

**Example**

```bash
cd 4.kmer_selector
python kmer_selector.py project [options]
```

### Purpose

Selects informative k-mers for each hierarchical node and trains decision-tree classifiers (optionally complemented by neural-network models).

### Main arguments

| Argument | Default |
|----------|---------|
| `project` | example |
| `-i`, `--input` | input |
| `-o`, `--output` | output |
| `-t`, `--tmp_path` | tmp |
| `-x`, `--min_k` | 4 |
| `-y`, `--max_k` | 8 |
| `--top_selected` | 100 |
| `--min_selected` | 10 |
| `--diverse_border` | 0.5 |
| `--common_border` | 0.8 |
| `--level_increment` | 0.2 |
| `-n`, `--use_NN` | True |
| `-m`, `--mmcs` | 30 |

Classifier models are written as PKL files together with text representations of decision trees.

---

# Step 5 — Generate Pseudo-Reads

**Program**

```
5.read_generator/generate_reads.py
```

**Example**

```bash
cd 5.read_generator
python generate_reads.py project_folder fasta [options]
```

### Purpose

Generates artificial sequencing reads from reference genomes for testing classifier performance.

### Required arguments

| Argument | Description |
|----------|-------------|
| `project_folder` | Input genome folder |
| `output_format` | `fasta` or `fastq` |

### Main optional arguments

| Argument | Default |
|----------|---------|
| `-n`, `--read_number` | 10000 |
| `--min_length` | 5000 |
| `--max_length` | 20000 |
| `--species_number` | 0 |
| `--seed` | random |

**Notes**

- Reads wrap around circular genomes.
- FASTQ qualities are randomly generated between Q20 and Q40.

---

# Step 6 — Classify Unknown Reads

**Program**

```
6.Classifier/identify_reads.py
```

**Example**

```bash
cd 6.Classifier
python identify_reads.py -f model.pkl -r reads.fasta -p project
```

### Purpose

Classifies unknown sequencing reads using the decision-tree model generated in Step 4.

### Required arguments

| Argument | Description |
|----------|-------------|
| `-f`, `--db_file` | Trained classifier model (.pkl) |
| `-r`, `--reads` | Input FASTA/FASTQ reads |

### Main optional arguments

| Argument | Default |
|----------|---------|
| `-i`, `--input` | input |
| `-o`, `--output` | output |
| `-d`, `--db_folder` | db |
| `-p`, `--project` | automatic |
| `-m`, `--min_read_length` | 5000 |
| `-x`, `--max_read_length` | unlimited |
| `-a`, `--accuracy` | 0.65 |
| `-s`, `--specificity` | 0.8 |
| `-n`, `--use_NN` | True |
| `-e`, `--entropy_cutoff` | 0.5 |

---

# Pipeline Summary

| Step | Description |
|------|-------------|
| **1** | Build k-mer frequency database (WDB) |
| **2** | Cluster genomes |
| **3** | Organize WDBs into hierarchical groups |
| **4** | Select informative k-mers and train classifiers |
| **5** | Generate pseudo-reads |
| **6** | Classify unknown reads |

---

## Repository Structure

```
1.seq2wdb/
2.create_tree/
3.WDB_sorter/
4.kmer_selector/
5.read_generator/
6.Classifier/
```

Each module can be executed independently, while together they form a complete pipeline for k-mer-based taxonomic identification.
