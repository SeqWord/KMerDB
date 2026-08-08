import sys, os, copy 
import numpy as np
import bitwiser, tools
from datetime import datetime
import container, nwmapper, progressbar, processor
from typing import List
from itertools import groupby
    
###############################################################################
class WordDB(container.Collection):
    def __init__(self, min_k: int = 0, max_k: int = 0, title: str = "", date: str = "", version: str = "1.0", qualifiers: dict | None = None):
        super().__init__(title = title, version = version, date = date)
        
        # Validate and store the k-mer range
        self.min_k = min_k
        self.max_k = max_k
        self.min_k, self.max_k = self._ascertain_range_borders(self.min_k, self.max_k, low_cutoff=-1)
        
        # Any additional parameters    
        self.qualifiers = self.para
        if qualifiers is not None:
            self.qualifiers.update(qualifiers)
                
        self.db = self.container
        
    def __add__(self, other):
        if not isinstance(other, WordDB):
            sys.exit(f"\n❌ Unsupported type of the additive {type(other).__name__}!")
        if self.min_k != other.min_k or self.max_k != other.max_k:
            sys.exit(f"\n❌ K-mer ranges of additives,[{self.min_k}..{self.max_k}] and [{other.min_k}..{other.max_k}], do not correspond each other!")
        oDB_copy = self.copy()
        for oGenome in other:
            oDB_copy.append(oGenome)
        return oDB_copy
        
    #### PUBLIC METHODS
    
    def add_genome(self, input_path: str, 
            targer_seq_length: int = 0, 
            chunk_number: int = 0, 
            flg_concatenate: bool = True,
            flg_keep_sequence: bool = False):
        # Get ID
        ID = self.get_next_ID()
        genome = tools.openSeqFile(input_path, flg_concatenate)
        genome_title, data = list(genome.items())[0]
        sequence = data["seq"]
        lineage = data["lineage"]
        
        oGenome = Genome(title = genome_title, ID = ID, min_k = self.min_k, max_k = self.max_k, lineage=lineage)
        
        tools.msg(genome_title)
        
        oGenome.process_sequence(sequence = sequence.upper(), 
            targer_seq_length = targer_seq_length, 
            chunk_number = chunk_number, 
            flg_keep_sequence = flg_keep_sequence)
        self.append(oGenome)

    def deplete_sequence(self, 
            in_path: str, 
            chunk_number: int, 
            min_chunk_length: int, 
            max_chunk_length: int,
            flg_keep_sequence: bool = False,
            flg_concatenate: bool = True,
        ):
        
        genome = tools.openSeqFile(in_path, flg_concatenate)
        genome_title, data = list(genome.items())[0]
        sequence = data["seq"]
        lineage = data["lineage"]
        # Fragment sequence to chunks
        chunks = tools.deplete_sequence(sequence=sequence, 
            chunk_number=chunk_number, 
            min_chunk_length=min_chunk_length, 
            max_chunk_length=max_chunk_length
        )
        
        bar = progressbar.indicator(len(chunks), f"{genome_title} processing: ")
        for i in range(len(chunks)):
            chunk = chunks[i]
            ID = self.get_next_ID()
            oGenome = Genome(title = genome_title, ID = ID, min_k = self.min_k, max_k = self.max_k, lineage=lineage)
        
            oGenome.process_sequence(sequence = chunk.upper(), flg_keep_sequence = flg_keep_sequence, echo=False)
            self.append(oGenome)
            bar(i + 1)
        bar.stop()
    
    # Create a matrix of requested genome sequence parameters
    def get_genome_parameter_matrix(
            self,
            gc_content: bool = False,
            gc_skew: bool = False,
            abs_gc_skew: bool = False,
            at_skew: bool = False,
            abs_at_skew: bool = False,
            purine_skew: bool = False,
            abs_purine_skew: bool = False,
            pattern_skew: bool = False,
            pattern_variance: bool = False,
            pattern_stdev: bool = False,
        ):
        """
        Create a matrix containing selected genome statistics.
    
        Each row represents one genome. The first row contains column
        headings. Parameters whose corresponding argument is False are
        omitted from the matrix.
    
        Returns
        -------
        list[list] or None
            Matrix in the form:
    
                [
                    ["Genome", "GC-content", "GC-skew", ...],
                    ["Genome_1", value1, value2, ...],
                    ["Genome_2", value1, value2, ...],
                    ...
                ]
    
            Returns None if no parameters were requested.
        """
    
        # Flags defining which genome statistics should be calculated.
        settings = [
            gc_content,
            gc_skew,
            abs_gc_skew,
            at_skew,
            abs_at_skew,
            purine_skew,
            abs_purine_skew,
            pattern_skew,
            pattern_variance,
            pattern_stdev,
        ]
    
        # Nothing was requested.
        if all(setting is False for setting in settings):
            tools.msg(
                "None of the genome statistic parameters were requested!"
            )
            return None
    
        # Column titles corresponding to the entries in `settings`.
        titles = [
            "GC-content",
            "GC-skew",
            "Abs-GC-skew",
            "AT-skew",
            "Abs-AT-skew",
            "Purine-skew",
            "Abs-Purine-skew",
            "Pattern-skew",
            "Pattern-variance",
            "Pattern-stdev",
        ]
    
        # Methods corresponding to the entries in `settings`.
        #
        # These are stored as method names rather than bound methods because
        # each function must later be called for a different Genome object.
        function_names = [
            "get_gc_content",
            "get_gc_skew",
            "get_abs_gc_skew",
            "get_at_skew",
            "get_abs_at_skew",
            "get_purine_skew",
            "get_abs_purine_skew",
            "get_pattern_skew",
            "get_pattern_variance",
            "get_pattern_std",
        ]
    
        # Create the header row containing only requested parameters.
        headings = ["Genome"]
    
        for i in range(len(settings)):
            if settings[i] is not False:
                headings.append(titles[i])
    
        # Initialize the matrix with the header row.
        matrix = [headings]
    
        # Process all genomes in the database.
        for oGenome in self:
            row = [oGenome.title]
    
            # Calculate only requested statistics.
            for i in range(len(settings)):
                if settings[i] is not False:
                    function = getattr(
                        oGenome,
                        function_names[i],
                    )
    
                    row.append(function())
    
            matrix.append(row)
    
        return matrix        

    def get_matrix(self,
            min_k: int = 0,
            max_k: int = 0,
            start_genome: int | str = 0,
            end_genome: int | str = 0,
            genome_list: list = [],
            label_taxon: str = "",                      # '' | species | genus
            data_type: str = "digit",                   # digit | count | z-score | median_centered-z-score
            digit_map: list = [-2, -1, 1, 2],           # four member list, aplicable for datatype 'digit' to transform '000', '001', '011', '111' to numbers
            matrix_geometry: str = "whole",             # whole | upper | lower
            out_file: str = "",                         # optional output file name
            kmer_title: str = "word"                    # word | triplet | combined = TTAA | 4,1,1 | TTAA,4,1,1
        ):

        """
        Create a matrix of k-mer values for selected genomes.
    
        The first row contains k-mer names. Each subsequent row contains
        the genome title followed by its k-mer values.
    
        Parameters
        ----------
        min_k, max_k : int
            Requested k-mer range. Values of 0 use the database limits.
    
        start_genome, end_genome : int or str
            Inclusive genome-index range. Genome titles may also be used.
            If end_genome is None, the last genome is used.
    
        genome_list : list, optional
            Explicit list of genome indices or titles. When provided, it
            takes precedence over start_genome and end_genome.
    
        label_taxon : str = ""
            One of "", "species", or "genus"
            Add additional 'Label' column after genome accession for grouping
    
        data_type : str
            One of "digit", "count", or "z-score".
    
        digit_map : list, optional
            Four values used to translate digital states in this order:
            000, 001, 011, 111. The default is [-2, -1, 1, 2].
    
        matrix_geometry : str
            Matrix output geometry:
                "whole" -> complete list of k-mers
                "upper" -> part of k-mers with x >= y
                "lower" -> part of k-mers with x <= y
    
        out_file : str
            Optional tab-delimited output filename.
    
        Returns
        -------
        list
            Matrix represented as a list of rows.
        """

        min_k, max_k = self._ascertain_range_borders(min_k, max_k)
        # matrix is requested from database subset, a database copy is created for the requested k-mer range
        if min_k != self.min_k or max_k != self.max_k:
            oWDB = self.copy(min_k=min_k, max_k=max_k)
            return oWDB.get_matrix(
                min_k=min_k,
                max_k=max_k,
                start_genome=start_genome,
                end_genome=end_genome,
                genome_list=genome_list,
                label_taxon=label_taxon,
                data_type=data_type,
                digit_map=digit_map,
                matrix_geometry=matrix_geometry,
                out_file=out_file,
                kmer_title=kmer_title
            )
        
        # Digit translator
        if len(digit_map) == 4:
            digit_map = dict(zip(['000', '001', '011', '111'], digit_map))
        elif len(digit_map) == 0:
            digit_map = {}
        else:
            raise ValueError(
                "\n❌ Digit_map must contain exactly four values "
                "corresponding to '000', '001', '011', and '111'."
            )
            
        # ------------------------------------------------------------
        # Determine which genomes should be included
        # ------------------------------------------------------------
    
        if genome_list:
            selected_genomes = genome_list
    
        else:
            # Convert a genome title into its integer index when necessary.
            if isinstance(start_genome, str):
                start_index = self.index(start_genome)
            else:
                start_index = start_genome
    
            # By default, continue to the final genome in the database.
            if end_genome == 0:
                end_index = len(self) - 1
            elif isinstance(end_genome, str):
                end_index = self.index(end_genome)
            else:
                end_index = end_genome
    
            if start_index is None:
                sys.exit(f"\n❌ Genome {start_genome} was not found!")
    
            if end_index is None:
                sys.exit(f"\n❌ Genome {end_genome} was not found!")
    
            try:
                start_index = int(start_index)
                end_index = int(end_index)
            except (ValueError, TypeError):
                sys.exit(
                    "\n❌ Start and end genome indices must be integers "
                    "or valid genome titles."
                )
    
            # Ensure that the lower index comes first.
            start_index, end_index = sorted(
                (start_index, end_index)
            )
    
            # Restrict the selected interval to valid database indices.
            start_index = max(0, start_index)
            end_index = min(len(self) - 1, end_index)
    
            selected_genomes = range(
                start_index,
                end_index + 1,
            )
    
        # ------------------------------------------------------------
        # Create the matrix
        # ------------------------------------------------------------
    
        matrix = []
        for genome_identifier in selected_genomes:
            # genome_identifier may be an integer index or genome title.
            oGenome = self[genome_identifier]
    
            if oGenome is None:
                tools.msg(
                    f"\n❌ Genome {genome_identifier} was not found!"
                )
                continue
    
            # get_values() returns records such as:
            #
            #     ATG,3,3,5,011
            #     ATG,3,3,5,14
            #     ATG,3,3,5,-1.276
            #
            kmer_records = oGenome.get_values(
                data_type=data_type,
                flg_add_kmers=True,
            )
    
            titles = []
            values = []
    
            for record in kmer_records:
                parts = record.split(",")
                # With flg_add_kmers=True, the first field is the
                # nucleotide representation of the k-mer.
                if kmer_title == "word":
                    titles.append(parts[0])
                elif kmer_title == "triplet":
                    titles.append(",".join(parts[1:-1]))
                elif kmer_title == "combined":
                    titles.append(",".join(parts[:-1]))
    
                raw_value = parts[-1]
    
                if data_type == "digit":
                    if raw_value not in digit_map:
                        raise ValueError(
                            f"\n❌ Unexpected digital value '{raw_value}' "
                            f"for k-mer {parts[0]}."
                        )
    
                    values.append(digit_map[raw_value])
    
                elif data_type == "count":
                    try:
                        values.append(int(raw_value))
                    except ValueError as exc:
                        raise ValueError(
                            f"\n❌ Invalid count '{raw_value}' "
                            f"for k-mer {parts[0]}."
                        ) from exc
    
                else:  # z-score
                    try:
                        values.append(float(raw_value))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid z-score '{raw_value}' "
                            f"for k-mer {parts[0]}."
                        ) from exc
    
            # Add the header before the first successfully processed genome.
            j = 2 if label_taxon else 1
            if not matrix:
                if label_taxon:
                    matrix.append(["Genome", "Label"] + titles)
                else:
                    matrix.append(["Genome"] + titles)
    
            # Ensure all genomes generated the same ordered k-mer set.
            elif matrix[0][j:] != titles:
                raise ValueError(
                    f"\n❌ K-mer order for genome '{oGenome.title}' "
                    "does not match the matrix header."
                )
            # Add label column if requested
            if label_taxon:
                matrix.append([oGenome.title, oGenome.get_taxon_label(label_taxon)] + values)
            else:
                matrix.append([oGenome.title] + values)
    
        # ------------------------------------------------------------
        # Optionally save the matrix
        # ------------------------------------------------------------
    
        if out_file:
            out_file = out_file.strip()
            delimeter = "\t"
            if out_file.lower().endswith(".csv"):
                delimeter = ","

            folder_path = os.path.dirname(
                os.path.abspath(out_file)
            )
    
            if not os.path.isdir(folder_path):
                sys.exit(
                    f"\n❌ Directory {folder_path} does not exist!"
                )
    
            data = "\n".join(
                delimeter.join(str(value) for value in row)
                for row in matrix
            )
    
            tools.saveTextFile(
                strText = data,
                fname=out_file,
            )
            tools.msg(f"✅ Matrix was saved to file {out_file}!")
    
        return matrix

        
    def get_distance_matrix(self,
        min_k: int = 0,
        max_k: int = 0,
        start_genome: int | str = 0,
        end_genome: int | str = 0,
        genome_list: list | None = None,
        distance_type: str = "hamming",             # hamming | rank | euclidean
        data_type: str = "median_centered-z-score", # count | z-score | median_centered-z-score
        matrix_geometry: str = "whole",             # whole | upper | lower
        out_file: str = "",                         # optional output filename
    ):
        """
        Create a distance matrix between genomes.
    
        The first row and first column contain genome titles.
    
        Parameters
        ----------
        min_k, max_k : int
            Requested k-mer range. Values of 0 use the database limits.
    
        start_genome, end_genome : int or str
            Inclusive genome range. Values may be integer indices or genome
            titles. If end_genome is 0, the last genome is selected.
    
        genome_list : list, optional
            Explicit list of genome indices or titles. When supplied, it
            takes precedence over start_genome and end_genome and must
            contain at least two genomes.
    
        distance_type : str
            Distance calculation:
                "hamming"   -> genome_a ^ genome_b
                "rank"      -> genome_a | genome_b
                "euclidean" -> genome_a & genome_b
    
        matrix_geometry : str
            Matrix output geometry:
                "whole" -> complete symmetric matrix
                "upper" -> diagonal and values above it
                "lower" -> diagonal and values below it
    
        out_file : str
            Optional tab-delimited output filename.
    
        Returns
        -------
        list
            Distance matrix represented as a list of rows.
        """
    
        if genome_list is None:
            genome_list = []
    
        distance_type = distance_type.lower()
        matrix_geometry = matrix_geometry.lower()
    
        allowed_distances = {"hamming", "rank", "euclidean"}
        allowed_geometries = {"whole", "upper", "lower"}
    
        if distance_type not in allowed_distances:
            raise ValueError(
                f"\n❌ Unsupported distance type '{distance_type}'. "
                f"Allowed values are: {', '.join(sorted(allowed_distances))}."
            )
    
        if matrix_geometry not in allowed_geometries:
            raise ValueError(
                f"\n❌ Unsupported matrix geometry '{matrix_geometry}'. "
                f"Allowed values are: {', '.join(sorted(allowed_geometries))}."
            )
    
        # Validate and normalize the requested k-mer range.
        min_k, max_k = self._ascertain_range_borders(min_k, max_k)
    
        # If only a subset of the stored k-mer range is requested,
        # construct a restricted database and generate the matrix from it.
        if min_k != self.min_k or max_k != self.max_k:
            oWDB = self.copy(min_k=min_k, max_k=max_k)
    
            return oWDB.get_distance_matrix(
                min_k=min_k,
                max_k=max_k,
                start_genome=start_genome,
                end_genome=end_genome,
                genome_list=genome_list,
                distance_type=distance_type,
                matrix_geometry=matrix_geometry,
                out_file=out_file,
            )
    
        # ------------------------------------------------------------
        # Determine which genomes should be included
        # ------------------------------------------------------------
    
        if genome_list:
            if len(genome_list) < 2:
                raise ValueError(
                    "\n❌ genome_list must contain at least two genomes."
                )
    
            selected_identifiers = genome_list
    
        else:
            # Convert the start genome title to its integer index.
            if isinstance(start_genome, str):
                start_index = self.index(start_genome)
            else:
                start_index = start_genome
    
            # A zero end value selects the final genome.
            if end_genome == 0:
                end_index = len(self) - 1
            elif isinstance(end_genome, str):
                end_index = self.index(end_genome)
            else:
                end_index = end_genome
    
            if start_index is None:
                raise ValueError(
                    f"\n❌ Genome '{start_genome}' was not found."
                )
    
            if end_index is None:
                raise ValueError(
                    f"\n❌ Genome '{end_genome}' was not found."
                )
    
            try:
                start_index = int(start_index)
                end_index = int(end_index)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "\n❌ Start and end genome indices must be integers "
                    "or valid genome titles."
                ) from exc
    
            # Put the smaller index first.
            start_index, end_index = sorted(
                (start_index, end_index)
            )
    
            # Restrict the interval to valid database indices.
            start_index = max(0, start_index)
            end_index = min(len(self) - 1, end_index)
    
            if start_index > end_index:
                raise ValueError(
                    "\n❌ The selected genome range is empty."
                )
    
            selected_identifiers = range(
                start_index,
                end_index + 1,
            )
    
        # ------------------------------------------------------------
        # Retrieve genome objects
        # ------------------------------------------------------------
    
        selected_genomes = []
    
        for identifier in selected_identifiers:
            try:
                oGenome = self[identifier]
            except (IndexError, KeyError, TypeError):
                oGenome = None
    
            if oGenome is None:
                tools.msg(f"Genome '{identifier}' was not found!")
                continue
    
            selected_genomes.append(oGenome)
    
        if len(selected_genomes) < 2:
            raise ValueError(
                "\n❌ At least two valid genomes are required to create "
                "a distance matrix."
            )
    
        titles = [oGenome.title for oGenome in selected_genomes]
        n_genomes = len(selected_genomes)
    
        # ------------------------------------------------------------
        # Select the comparison operator
        # ------------------------------------------------------------
    
        def calculate_distance(genome_a, genome_b):
            if distance_type == "hamming":
                return genome_a ^ genome_b
    
            if distance_type == "rank":
                return genome_a | genome_b
    
            # distance_type == "euclidean"
            return genome_a & genome_b
    
        # ------------------------------------------------------------
        # Calculate all pairwise distances
        # ------------------------------------------------------------
    
        # First create a complete numeric matrix. This avoids performing
        # the same symmetric comparison twice.
        distances = [
            [0.0 for _ in range(n_genomes)]
            for _ in range(n_genomes)
        ]
    
        for i in range(n_genomes):
            for j in range(i + 1, n_genomes):
                distance = calculate_distance(
                    selected_genomes[i],
                    selected_genomes[j],
                )
    
                distances[i][j] = distance
                distances[j][i] = distance
    
        # ------------------------------------------------------------
        # Format the requested matrix geometry
        # ------------------------------------------------------------
    
        d_matrix = [["Genome"] + titles]
    
        for i, title in enumerate(titles):
            row = [title]
    
            for j in range(n_genomes):
                if matrix_geometry == "whole":
                    value = distances[i][j]
    
                elif matrix_geometry == "upper":
                    # Retain the diagonal and cells above it.
                    value = distances[i][j] if j >= i else ""
    
                else:  # lower
                    # Retain the diagonal and cells below it.
                    value = distances[i][j] if j <= i else ""
    
                row.append(value)
    
            d_matrix.append(row)
    
        # ------------------------------------------------------------
        # Optionally save the matrix
        # ------------------------------------------------------------
    
        if out_file:
            folder_path = os.path.dirname(
                os.path.abspath(out_file)
            )
    
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"\n❌ Directory '{folder_path}' does not exist."
                )
    
            data = "\n".join(
                "\t".join(str(value) for value in row)
                for row in d_matrix
            )
    
            tools.saveTextFile(
                strText=data,
                fname=out_file,
            )
    
            tools.msg(
                f"✅ Matrix was saved to file {out_file}!"
            )
    
        return d_matrix
        
    def cluster_genomes(
        self,
        min_k: int = 0,
        max_k: int = 0,
        start_genome: int | str = 0,
        end_genome: int | str = 0,
        genome_list: list | None = None,
        distance_type: str = "hamming",     # hamming | rank | euclidean
        cl_algorithm: str = "spectral",     # upgma | nj | spectral
        output_format: str = "newick",      # pathways | newick
        out_file: str = "",                 # optional Newick filename
        echo: bool = True,                  # print saving file on the screen
    ):
        """
        Create a cluster tree from pairwise distances between genomes.
    
        Parameters
        ----------
        min_k, max_k : int
            Requested k-mer range. Values of 0 use the database limits.
    
        start_genome, end_genome : int or str
            Inclusive genome range. Values may be integer indices or genome
            titles. If end_genome is 0, the last genome is selected.
    
        genome_list : list, optional
            Explicit list of genome indices or titles. When supplied, it
            takes precedence over start_genome and end_genome and must
            contain at least three genomes.
    
        distance_type : str
            Distance calculation:
                "hamming"   -> genome_a ^ genome_b
                "rank"      -> genome_a | genome_b
                "euclidean" -> genome_a & genome_b
    
        cl_algorithm : str
            Clustering algorithm:
                "upgma"
                "nj"
                "spectral"
    
        output_format : str
            Currently applicable only for spectral clustering. Returns either newick tree or text cluster reptresentation
                "newick"
                "pathways"
    
        out_file : str
            Optional output filename. UPGMA and NJ save a Newick tree.
            The spectral implementation controls its output through its
            output_format and output_file arguments.
    
        Returns
        -------
        list
            Paths from the root to the terminal genome nodes, as returned
            by the selected clustering implementation.
        """
    
        # Import the tree-building functions from make_tree.py,
        # which must be available in the same import path.
        from make_tree import (
            upgma_paths_from_upper_triangle,
            nj_paths_from_upper_triangle,
            spa,
        )
    
        if genome_list is None:
            genome_list = []
    
        distance_type = distance_type.lower()
        cl_algorithm = cl_algorithm.lower()
    
        allowed_distances = {
            "hamming",
            "rank",
            "euclidean",
        }
    
        allowed_algorithms = {
            "upgma",
            "nj",
            "spectral",
        }
    
        if distance_type not in allowed_distances:
            raise ValueError(
                f"\n❌ Unsupported distance type '{distance_type}'. "
                f"Allowed values are: "
                f"{', '.join(sorted(allowed_distances))}."
            )
    
        if cl_algorithm not in allowed_algorithms:
            raise ValueError(
                f"\n❌ Unsupported clustering algorithm '{cl_algorithm}'. "
                f"Allowed values are: "
                f"{', '.join(sorted(allowed_algorithms))}."
            )
    
        # Validate and normalize the requested k-mer range.
        min_k, max_k = self._ascertain_range_borders(
            min_k,
            max_k,
        )
    
        # If only part of the database's k-mer range is requested,
        # create a restricted copy and perform clustering on that copy.
        if min_k != self.min_k or max_k != self.max_k:
            oWDB = self.copy(
                min_k=min_k,
                max_k=max_k,
            )
    
            return oWDB.cluster_genomes(
                min_k=min_k,
                max_k=max_k,
                start_genome=start_genome,
                end_genome=end_genome,
                genome_list=genome_list,
                distance_type=distance_type,
                cl_algorithm=cl_algorithm,
                out_file=out_file,
            )
    
        # ------------------------------------------------------------
        # Determine which genomes should be included
        # ------------------------------------------------------------
    
        if genome_list:
            if len(genome_list) < 3:
                raise ValueError(
                    "\n❌ genome_list must contain at least two genomes."
                )
    
            selected_identifiers = genome_list
    
        else:
            # Resolve the beginning of the requested genome range.
            if isinstance(start_genome, str):
                start_index = self.index(start_genome)
            else:
                start_index = start_genome
    
            # A zero end value means the final genome in the database.
            if end_genome == 0:
                end_index = len(self) - 1
            elif isinstance(end_genome, str):
                end_index = self.index(end_genome)
            else:
                end_index = end_genome
    
            if start_index is None:
                raise ValueError(
                    f"\n❌ Genome '{start_genome}' was not found."
                )
    
            if end_index is None:
                raise ValueError(
                    f"\n❌ Genome '{end_genome}' was not found."
                )
    
            try:
                start_index = int(start_index)
                end_index = int(end_index)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "\n❌ Start and end genome indices must be integers "
                    "or valid genome titles."
                ) from exc
    
            # Accept the boundaries in either order.
            start_index, end_index = sorted(
                (start_index, end_index)
            )
    
            # Restrict the interval to valid database positions.
            start_index = max(0, start_index)
            end_index = min(len(self) - 1, end_index)
    
            if start_index > end_index:
                raise ValueError(
                    "\n❌ The selected genome range is empty."
                )
    
            selected_identifiers = range(
                start_index,
                end_index + 1,
            )
    
        # ------------------------------------------------------------
        # Retrieve genome objects
        # ------------------------------------------------------------
    
        selected_genomes = []
    
        for identifier in selected_identifiers:
            try:
                oGenome = self[identifier]
            except (IndexError, KeyError, TypeError):
                oGenome = None
    
            if oGenome is None:
                tools.msg(
                    f"\n❌ Genome '{identifier}' was not found!"
                )
                continue
    
            selected_genomes.append(oGenome)
    
        if len(selected_genomes) < 2:
            raise ValueError(
                "\n❌ At least two valid genomes are required "
                "for clustering."
            )
    
        titles = [
            str(oGenome.title)
            for oGenome in selected_genomes
        ]
    
        # Duplicate labels are technically possible in Newick, but they
        # make interpretation of the resulting tree ambiguous.
        if len(set(titles)) != len(titles):
            raise ValueError(
                "\n❌ Genome titles must be unique for tree construction."
            )
    
        # ------------------------------------------------------------
        # Select the appropriate overloaded distance operator
        # ------------------------------------------------------------
    
        def calculate_distance(genome_a, genome_b):
            if distance_type == "hamming":
                return genome_a ^ genome_b
    
            if distance_type == "rank":
                return genome_a | genome_b
    
            # distance_type == "euclidean"
            return genome_a & genome_b
    
        # ------------------------------------------------------------
        # Construct an upper-triangular distance matrix
        # ------------------------------------------------------------
        #
        # For n genomes:
        #
        # [
        #     [d01, d02, d03, ...],
        #     [     d12, d13, ...],
        #     [          d23, ...],
        #     ...
        # ]
        #
        # The final genome has no genomes to its right, so the matrix
        # contains n - 1 rows.
    
        d_matrix = []
    
        for i in range(len(selected_genomes) - 1):
            row = []
    
            for j in range(i + 1, len(selected_genomes)):
                distance = calculate_distance(
                    selected_genomes[i],
                    selected_genomes[j],
                )
    
                try:
                    distance = float(distance)
                except (ValueError, TypeError) as exc:
                    raise TypeError(
                        f"\n❌ Distance between genomes '{titles[i]}' and "
                        f"'{titles[j]}' is not numeric: {distance!r}."
                    ) from exc
    
                if not np.isfinite(distance):
                    raise ValueError(
                        f"\n❌ Distance between genomes '{titles[i]}' and "
                        f"'{titles[j]}' is not finite: {distance}."
                    )
    
                if distance < 0:
                    raise ValueError(
                        f"\n❌ Distance between genomes '{titles[i]}' and "
                        f"'{titles[j]}' is negative: {distance}."
                    )
    
                row.append(distance)
    
            d_matrix.append(row)
    
        # ------------------------------------------------------------
        # Validate output directory
        # ------------------------------------------------------------
    
        if out_file:
            folder_path = os.path.dirname(
                os.path.abspath(out_file)
            )
    
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"Directory '{folder_path}' does not exist."
                )
    
        # ------------------------------------------------------------
        # Construct the tree
        # ------------------------------------------------------------
    
        if cl_algorithm == "upgma":
            pathways = upgma_paths_from_upper_triangle(
                tri=d_matrix,
                labels=titles,
                output_file=out_file,
            )
    
        elif cl_algorithm == "nj":
            pathways = nj_paths_from_upper_triangle(
                tri=d_matrix,
                labels=titles,
                output_file=out_file,
            )
    
        else:
            # These values reproduce the defaults used by execute()
            # in the attached make_tree module.
            pathways = spa(
                matrix=d_matrix,
                labels=titles,
                max_cluster_number=5,
                max_cluster_content=5,
                max_levels=5,
                force_k=0,
                output_format=output_format,
                output_file=out_file,
            )
    
        if out_file and echo:
            tools.msg(
                f"✅ Cluster tree was saved to file {out_file}!"
            )
    
        return pathways
            
    def clear(self):
        self.qualifiers = {}
        if len(self.container):
            for i in range(len(self.container) - 1, -1, -1):
                del self[i]
            
    def save_dbfile(self, path: str, echo: bool = True):
        directory = os.path.dirname(os.path.abspath(path))
    
        if not os.path.isdir(directory):
            sys.exit(f"\n❌ Directory {directory} does not exist!")
        try:
            tools.saveDBFile(
                fname=path,
                data=self.copy(),
            )
            if echo:
                tools.msg(f"✅ Database file {path} with k-mer counts of {len(self)} genomes was successfuly saved!")
        except:
            tools.msg(f"\n❌ Error occures when database file {path} was saving!")
    
    def open_dbfile(self, path=""):
        if not os.path.exists(path):
            sys.exit(f"\n❌ Database file {path} does not exist!")
        fname,DB,supplementary = tools.openDBFile(path)
        # Set k-mer range if an empty DB container was created
        if self.min_k == 0:
            self.min_k = DB.min_k
        if self.max_k == 0:
            self.max_k = DB.max_k
        DB = DB.copy(min_k=self.min_k, max_k=self.max_k)
        self.title = DB.title
        self.version = DB.version
        self.date = DB.date 
        
        # Prevent loaded records from being added to existing records.
        self.clear()
        
        if DB.qualifiers:
            self.qualifiers.update(DB.qualifiers)
        for oGenome in DB:
            self.append(oGenome)
        
    def copy(self, min_k: int = 0, max_k: int = 0):
        min_k, max_k = self._ascertain_range_borders(min_k, max_k)
                    
        oNewDB = WordDB(min_k=min_k, max_k=max_k, title=self.title, version=self.version)
        oNewDB.qualifiers.update(self.qualifiers)
        for oGenome in self:
            oNewGenome = oGenome.copy(min_k, max_k)
            oNewDB.append(oNewGenome)
        return oNewDB

    #### INFO
    def get_info(self):
        output = []
        output.append(f"Database '{self.title}' version {self.version}")
        output.append("Genomes:\t" + str(len(self.container)))
        return "\n".join(output)
        
    #### PRIVATE METHODS
    def _ascertain_range_borders(self, min_k: int, max_k: int, low_cutoff: int = 0):
        if min_k == 0:
            min_k = self.min_k
        if max_k == 0:
            max_k = self.max_k
        try:
            min_k, max_k = [int(v) for v in [min_k, max_k]]
        except:
            sys.exit("\n❌ Error copying WordDB: values min_k and max_k must be integers!")
        
        tools.ascertain_range_borders(min_k, max_k + 1, low_cutoff)
        return min_k, max_k
        
###############################################################################
class Genome:
    def __init__(self, 
        title: str, 
        min_k: int, 
        max_k: int, 
        ID: int, 
        accession: str = "", 
        lineage: str = "", 
        sequence: str = "", 
        qualifiers: dict = {}):
            
        self.title = title
        self.min_k = min_k
        self.max_k = max_k
        self.ID = ID
        self.accession = accession
        self.lineage = lineage
        self.sequence = ""
        self.seqlength = len(sequence)          # Sequence length
        self.string = self._generate_string()   # Integer representation of word (k-mer) counts represented by 3 digits: '000', '001', '011', '111' 
        self.word_counts = {}                   # Word counts
        self.qualifiers = qualifiers
        self.oMapper = nwmapper.Mapper()
        self.ATGC = {}
        # Processing sequence string by calling 'process_sequence' on later stage is more preferable as this function provided more options
        if sequence:
            self.process_sequence(sequence)
            
    #### COMPARISON OPERATORS
    
    # Operator &: Euclidean distance between k-mer z-score vectors
    def __and__(self, other):
        return self._euclidean_distance(other)
        
            
    # Operator ^: normalized Hamming distance between digital strings
    def __xor__(self, other):
        return self._hamming_distance(other)
        

    # Operator |: compare k-mer z-score ranks
    def __or__(self, other):
        return self._rank_distance(other)
    
            
    #### PUBLIC METHODS
    
    # generate list of words in formats 'wlength,x,y' or 'kmer,wlength,x,y'
    def generate_kmers(self, min_k: int = 0, max_k: int = 0, flg_add_kmers: bool = False):
        # Cgeck proper setting of k-mer borders
        if any([min_k, max_k]):
            tools.ascertain_range_borders(min_k, max_k + 1)
        else:
            min_k = self.min_k
            max_k = self.max_k
        kmers = []
        for k in range(min_k, max_k + 1):
            if flg_add_kmers:
                kmers += [",".join([self.oMapper(w,x,y),str(w),str(x),str(y)]) for w,x,y in self.oMapper.generate(k)]
            else:
                kmers += [",".join([str(w),str(x),str(y)]) for w,x,y in self.oMapper.generate(k)]
        return kmers

    def get_kmer_value(self, kmer: str | list):
        if isinstance(kmer, str):   # Like 'ATGC'
            wl, x, y = self.oMapper(kmer)
        elif isinstance(kmer, list):
            if len(kmer) == 4:
                wl, x, y = kmer[1:]
            elif len(kmer) == 3:
                wl, x, y = kmer
            else:
                sys.exit(f"\n❌ Unsupported k-mer format: {kmer}!")
        else:
            sys.exit(f"\n❌ Unsupported k-mer format: {kmer}!")
        try:
            wl, x, y = [int(v) for v in [wl, x, y]]
        except:
            sys.exit(f"\n❌ Unsupported k-mer format: {kmer}!")
        if self.has(wl, x, y):
            return self.word_counts[wl]['x'][x][y]
        else:
            return None

    def get_kmer_by_index(self, index: int, flg_add_kmers: bool = False):
        kmers = self.generate_kmers(flg_add_kmers = flg_add_kmers)
        if -1 < index < len(kmers):
            return kmers[index]
        return None

    def add(self, wlength: int, x: int, y: int, count: int = 0, echo: bool = False):
        # Check if such word already exists and sum counts
        if self.has(wlength,x,y):
            self.word_counts[wlength]['x'][x][y] += count
        if wlength not in self.word_counts:
            self.word_counts[wlength] = {'x':{}}
        if x not in self.word_counts[wlength]['x']:
            self.word_counts[wlength]['x'][x] = {y:count}
        else:
            self.word_counts[wlength]['x'][x][y] = count
        
        value = self._transform_to_digital_value(wlength=wlength, count=count, seqlength=self.seqlength, binary_code=True)
        word = f"{wlength},{x},{y}" 
        self._set_digital_value(word=word, value=value)
            
        return True

    # Counting kmers and populating the database
    def process_sequence(self, sequence: str, 
        targer_seq_length: int = 0, 
        chunk_number: int = 0, 
        flg_keep_sequence: bool = False, 
        flg_reset_sequence: bool = False,
        echo: bool = True):
            
        # Check if sequence reset was requested. self.ATGC is used as a marker that one sequence was processed recently
        if self.ATGC and self.sequence and not flg_reset_sequence:
            sys.exit("\n❌ The existing sequence cannot be reset without a special requisition!")
            
        if flg_keep_sequence:
            self.sequence = sequence
        
        # Process sequence
        if targer_seq_length > 0 and chunk_number > 0:
            sequence = self._deplete_sequence(sequence, targer_seq_length, chunk_number)
        # Reset sequence if requested
        if flg_keep_sequence:
            self.sequence = sequence
            
        self.seqlength = len(sequence)
        
        for wlength in range(self.min_k,self.max_k + 1):
            self._count_words(sequence=sequence, 
                wlength=wlength, 
                flg_progress=echo, bar_text="Count "+str(wlength)+"-mers: ")
            words = [kmer.split(",") for kmer in self.generate_kmers(min_k=wlength, max_k=wlength)]
            length = len(words)
            
            if echo:
                bar = progressbar.indicator(length, str(wlength)+"-mers stat: ")
            counter = 1
            for wl, x, y in words:
                if not self.has(wl,x,y):
                    continue
                count = self.get_kmer_value(wl,x,y)
                self.add(wl,x,y,count)
                counter += 1
                if echo and counter%99 == 0:
                    try:
                        bar(counter)
                    except:
                        pass
            if echo:
                bar.stop()
            
        self.ATGC = {"A":sequence.upper().count("A"),
            "T":sequence.upper().count("T"),
            "G":sequence.upper().count("G"),
            "C":sequence.upper().count("C"),
        }

        
    def get_values(
        self,
        min_k: int = 0,
        max_k: int = 0,
        data_type: str = "digit",           # digit | count | z-score | median_centered-z-score
        matrix_geometry: str = "whole",     # whole | upper | lower
        flg_add_kmers: bool = False,
        flg_reverse_complement: bool = False
    ):
        """
        Return k-mer values in one of the following formats:
    
            digit   - three-bit digital value: 000, 001, 011, or 111
            count   - observed k-mer count
            z-score - standardized deviation from the expected count
    
        Each returned element contains the k-mer information generated by
        `generate_kmers()`, followed by the requested value.
    
        Parameters
        ----------
        min_k : int, default=0
            Minimum k-mer length. A value of 0 uses self.min_k.
    
        max_k : int, default=0
            Maximum k-mer length. A value of 0 uses self.max_k.
    
        data_type : str, default="digit"
            Output value type: "digit", "count", or "z-score".
    
        matrix_geometry : str
            Matrix output geometry:
                "whole" -> complete list of k-mers
                "upper" -> part of k-mers with x >= y
                "lower" -> part of k-mers with x <= y
    
        flg_add_kmers : bool, default=False
            Passed to generate_kmers(). Determines whether nucleotide
            representations are included with the k,x,y indices.
    
        Returns
        -------
        list[str]
            List of comma-separated k-mer records and their values.
        """
    
        # Use the complete stored range when limits are not specified.
        if min_k == 0:
            min_k = self.min_k
    
        if max_k == 0:
            max_k = self.max_k
    
        # Ensure that the requested range is within the stored range.
        if (
            min_k < self.min_k
            or max_k > self.max_k
            or min_k > max_k
        ):
            sys.exit(
                f"\n❌ Wrong k-mer range [{min_k}..{max_k}]! "
                f"The range must be within "
                f"[{self.min_k}..{self.max_k}] and min_k <= max_k."
            )
    
        # Normalize the format once instead of repeatedly calling lower().
        data_type = data_type.lower()
    
        allowed_formats = {"digit", "count", "z-score", "median_centered-z-score"}
    
        if data_type not in allowed_formats:
            sys.exit(
                f"\n❌ Unsupported value format '{data_type}'! "
                f"Allowed formats are: {', '.join(sorted(allowed_formats))}."
            )
    
        # Create the ordered list of k-mers for the requested range.
        #
        # Each item is expected to end with:
        #     k,x,y
        #
        # Examples:
        #     "3,3,5"
        #     "ATG,3,3,5"
        kmers = self.generate_kmers(
            min_k=min_k,
            max_k=max_k,
            flg_add_kmers=flg_add_kmers,
        )
    
        # ------------------------------------------------------------
        # Return three-bit digital abundance values
        # ------------------------------------------------------------
        if data_type == "digit":
    
            # Use the complete bit string when the complete k-mer range
            # was requested; otherwise extract only the requested subset.
            if min_k == self.min_k and max_k == self.max_k:
                digital_string = self.string
            else:
                digital_string = self.get_string_subset(
                    min_k=min_k,
                    max_k=max_k,
                )
    
            # bin() produces a string beginning with "0b1".
            # The leading 1 is the sentinel used to preserve leading zeros,
            # so remove all three characters: "0b1".
            bits = bin(digital_string)[3:]
    
            # Every k-mer is represented by exactly three consecutive bits.
            values = [
                f"{kmer},{bits[i * 3:i * 3 + 3]}"
                for i, kmer in enumerate(kmers)
            ]
            
            if flg_reverse_complement:
                values = [self._reverse_complement_kmer(value[:value.rfind(",")]) + value[value.rfind(","):] for value in values]
                
            if matrix_geometry.lower() == "upper":
                values = [value for value in values if self._is_upper(value.split(",")[:-1])]
            
            if matrix_geometry.lower() == "lower":
                values = [value for value in values if self._is_lower(value.split(",")[:-1])]
            
            return values
    
        # ------------------------------------------------------------
        # Return observed k-mer counts
        # ------------------------------------------------------------
        if data_type == "count":
            values = []
    
            for kmer in kmers:
                parts = kmer.split(",")
    
                try:
                    # This works for both "k,x,y" and "word,k,x,y".
                    k, x, y = [int(v) for v in parts[-3:]]
                except (ValueError, TypeError):
                    sys.exit(f"\n❌ Cannot parse k-mer record '{kmer}'.")
                    
                if matrix_geometry.lower() == "upper" and self._is_lower([k, x, y]):
                    continue
    
                if matrix_geometry.lower() == "lower" and self._is_upper([k, x, y]):
                    continue
    
                # Return zero when the k-mer is absent from word_counts.
                if flg_reverse_complement:
                    count = (
                        self.word_counts
                        .get(k, {})
                        .get("x", {})
                        .get(y, {})
                        .get(x, 0)
                    )
                else:
                    count = (
                        self.word_counts
                        .get(k, {})
                        .get("x", {})
                        .get(x, {})
                        .get(y, 0)
                    )
    
                values.append(f"{kmer},{count}")
    
            return values
            
        if data_type == "z-score":
            # ------------------------------------------------------------
            # Return z-scores of observed k-mer counts compared to expectations
            # ------------------------------------------------------------
            values = []
        
            # Number of canonical nucleotides A, T, G, and C.
            seqlength = self.get_seqlength(True)
        
            if seqlength <= 0:
                sys.exit(
                    "\n❌ The sequence contains no canonical A, T, G, or C "
                    "nucleotides."
                )
            '''
            # Retain this restriction if it is required by your statistical
            # model. Note that it becomes very stringent for large k values.
            min_required_length = 4 ** (2 * max_k)
        
            if seqlength < min_required_length:
                sys.exit(
                    "\n❌ Sequence length is too short for this calculation! "
                    f"Sequence length must be at least {min_required_length}; "
                    f"observed length is {seqlength}."
                )
            '''
            # Cache expectation and standard deviation because all k-mers of
            # the same length have the same values under the equal-frequency
            # random-sequence model.
            current_k = None
            expectation = 0.0
            sigma = 0.0
        
            for kmer in kmers:
                parts = kmer.split(",")
        
                try:
                    k, x, y = [int(v) for v in parts[-3:]]
                except (ValueError, TypeError):
                    sys.exit(f"\n❌ Cannot parse k-mer record '{kmer}'.")
    
                if matrix_geometry.lower() == "upper" and self._is_lower([k, x, y]):
                    continue
    
                if matrix_geometry.lower() == "lower" and self._is_upper([k, x, y]):
                    continue    
        
                # Recalculate the expected count and standard deviation only
                # when the k-mer length changes.
                if k != current_k:
                    current_k = k
        
                    # Number of possible starting positions of a k-mer.
                    n_positions = seqlength - k + 1
        
                    if n_positions <= 0:
                        sys.exit(
                            f"\n❌ Sequence length {seqlength} is shorter than "
                            f"k-mer length {k}."
                        )
        
                    # Under an equal-frequency independent-nucleotide model,
                    # the probability of a specific k-mer is 1 / 4**k.
                    expectation = n_positions / (4**k)
        
                    # Binomial standard deviation:
                    #
                    # sqrt[n * p * (1 - p)]
                    #
                    # where n = L-k+1 and p = 1/4**k.
                    sigma = np.sqrt(
                        n_positions
                        * (4**k - 1)
                        / (4 ** (2 * k))
                    )
        
                # Retrieve the observed count; use zero if the k-mer is absent.
                if flg_reverse_complement:
                    count = (
                        self.word_counts
                        .get(k, {})
                        .get("x", {})
                        .get(y, {})
                        .get(x, 0)
                    )
                else:
                    count = (
                        self.word_counts
                        .get(k, {})
                        .get("x", {})
                        .get(x, {})
                        .get(y, 0)
                    )
        
                # sigma should be positive when n_positions > 0, but this
                # check prevents an accidental division by zero.
                if sigma == 0:
                    z_score = 0.0
                else:
                    z_score = (count - expectation) / sigma
        
                values.append(f"{kmer},{z_score}")
        
            return values

        if data_type == "median_centered-z-score":
            # ------------------------------------------------------------
            # Return median-centered z-scores of observed k-mer counts
            # ------------------------------------------------------------
            values = []
            
            seqlength = self.get_seqlength(True)
            
            if seqlength <= 0:
                sys.exit(
                    "\n❌ The sequence contains no canonical A, T, G, or C "
                    "nucleotides."
                )
            
            # ------------------------------------------------------------
            # 1. Collect counts, grouped by k
            # ------------------------------------------------------------
            records = {}
            
            for kmer in kmers:
                parts = kmer.split(",")
            
                try:
                    k, x, y = [int(v) for v in parts[-3:]]
                except (ValueError, TypeError):
                    sys.exit(f"\n❌ Cannot parse k-mer record '{kmer}'.")
            
                if matrix_geometry.lower() == "upper" and self._is_lower([k, x, y]):
                    continue
            
                if matrix_geometry.lower() == "lower" and self._is_upper([k, x, y]):
                    continue
            
                if seqlength - k + 1 <= 0:
                    sys.exit(
                        f"\n❌ Sequence length {seqlength} is shorter than "
                        f"k-mer length {k}."
                    )
            
                if flg_reverse_complement:
                    count = (
                        self.word_counts
                        .get(k, {})
                        .get("x", {})
                        .get(y, {})
                        .get(x, 0)
                    )
                else:
                    count = (
                        self.word_counts
                        .get(k, {})
                        .get("x", {})
                        .get(x, {})
                        .get(y, 0)
                    )
            
                records.setdefault(k, []).append((kmer, count))
            
            
            # ------------------------------------------------------------
            # 2. Calculate z-scores independently for every k
            # ------------------------------------------------------------
            for k, items in records.items():
            
                counts = np.asarray(
                    [count for _, count in items],
                    dtype=float
                )
            
                # Median of the complete k-mer count distribution.
                median = np.median(counts)
            
                # Standard deviation calculated around the median,
                # rather than around the mean.
                sigma = np.sqrt(
                    np.mean((counts - median) ** 2)
                )
            
                if sigma == 0:
                    z_scores = np.zeros(len(counts), dtype=float)
            
                else:
                    # Median-centered scores.
                    z_scores = (counts - median) / sigma
            
                    # Force the complete distribution to have mean 0.
                    # Consequently sum(z_scores) ~= 0.
                    z_scores -= np.mean(z_scores)
            
                for (kmer, _), z_score in zip(items, z_scores):
                    values.append(f"{kmer},{z_score}")
            
            
            return values

        sys.exit(f"ERROR: Requested unknown data type {data_type}!")
            
    # Return a substring of values [min_k..max_k]
    def get_string_subset(self, min_k: int = 0, max_k: int = 0):
        if min_k == 0:
            min_k = self.min_k
        if max_k == 0:
            max_k = self.max_k
        if min_k == self.min_k and max_k == self.max_k:
            return self.string 
            
        if (
            min_k < self.min_k
            or max_k > self.max_k
            or min_k > max_k
        ):
            sys.exit(
                f"\n❌ Wrong min_k={min_k} and max_k={max_k} settings! "
                f"Values must be within "
                f"[{self.min_k}..{self.max_k}] and satisfy "
                f"min_k <= max_k."
            )
    
        bits = bin(self.string)[2:]          # remove '0b'
    
        left = 1 + 3 * sum(4**k for k in range(self.min_k, min_k))
        length = 3 * sum(4**k for k in range(min_k, max_k + 1))
    
        return int("1" + bits[left:left + length], 2)
        
    def get_taxon_label(self, taxon: str):
        if not self.lineage:
            return "ND"
        delimeters = ["|",",", ";",">"]
        lineage = ""
        for delimeter in delimeters:
            if self.lineage.count(delimeter) >= 4:
                lineage = [s.strip() for s in self.lineage.split(delimeter)]
                break
                
        if not lineage:
            return "ND"
            
        if taxon.lower() == "species":
            species_parts = lineage[-2].split()
            if len(species_parts) >= 2:
                species = " ".join(species_parts[:2])
            else:
                species = " ".join(species_parts)
            return species
        elif taxon.lower() == "genus":
            genus = lineage[-3]
            if genus.lower().find("complex") > -1 or genus.lower().find("group") > -1:
                genus = lineage[-4]
            return genus
        return "ND"
    
    def copy(
        self,
        min_k: int = 0,
        max_k: int = 0,
        reverse_complement: bool = False,
    ):
        """
        Create a copy of the genome.
    
        Optionally restrict the copy to a specified k-mer range and/or
        create the reverse-complement representation.
        """
    
        # No range specified: retain the complete stored range.
        if min_k == 0 and max_k == 0:
            min_k = self.min_k
            max_k = self.max_k
    
        # Reject incomplete range specification.
        elif min_k == 0 or max_k == 0:
            raise ValueError(
                "\n❌ Both min_k and max_k must be specified together."
            )
    
        # Ensure that the requested range is available.
        if min_k < self.min_k or max_k > self.max_k or min_k > max_k:
            raise ValueError(
                f"\n❌ The requested k-mer range [{min_k}..{max_k}] "
                f"is outside the available range "
                f"[{self.min_k}..{self.max_k}]."
            )
    
        oGenome = Genome(
            title=self.title,
            min_k=min_k,
            max_k=max_k,
            ID=self.ID,
            accession=self.accession,
            lineage=self.lineage,
        )
        oGenome.seqlength = self.seqlength
    
        # Retain only word-count records within the requested k-mer range.
        WordCounts = {
            wl: copy.deepcopy(data)
            for wl, data in self.word_counts.items()
            if min_k <= wl <= max_k
        }
    
        # Extract the corresponding portion of the digital string.
        if min_k == self.min_k and max_k == self.max_k:
            oGenome.string = self.string
        else:
            oGenome.string = self.get_string_subset(
                min_k=min_k,
                max_k=max_k,
            )
    
        oGenome.word_counts = WordCounts
    
        if self.ATGC:
            oGenome.ATGC.update(self.ATGC)
    
        # ------------------------------------------------------------
        # Create the reverse-complement representation
        # ------------------------------------------------------------
    
        if reverse_complement:
            original_ATGC = dict(oGenome.ATGC)
    
            # Remove the current k-mer counts and digital values before
            # reconstructing them in reverse-complement coordinates.
            oGenome.clean()
    
            for k, k_data in WordCounts.items():
                for x, x_data in k_data["x"].items():
                    for y, count in x_data.items():
                        rc_k, rc_x, rc_y = self._reverse_complement_kmer(
                            [k, x, y]
                        )
    
                        oGenome.add(
                            rc_k,
                            rc_x,
                            rc_y,
                            count,
                        )
    
            # Complementary nucleotide counts:
            # A <-> T and G <-> C.
            oGenome.ATGC = {
                "A": original_ATGC.get("T", 0),
                "T": original_ATGC.get("A", 0),
                "G": original_ATGC.get("C", 0),
                "C": original_ATGC.get("G", 0),
            }
    
        oGenome.qualifiers.update(self.qualifiers)
        oGenome.sequence = self.sequence
    
        return oGenome
            
    def clean(self):
        self.string = self._generate_string()
        self.word_counts = {}
        self.ATGC = {}
        self.qualifiers = {}
        self.sequence = ""
        
    #### INFO
    def has(self, wlength: int, x: int, y: int, flg_complement: bool = False):
        try:
            count = word_counts[wlength]['x'][x][y]
            return True
        except:
            if flg_complement:
                wlength,x,y = self.oMapper.revcomplement([wlength,x,y])
                return self.has(wlength,x,y)
            else:
                return False
                
    def get_seqlength(self, flg_canonical: bool = True):
        # Canonical nucleotides are A, T, G, and C
        if flg_canonical:
            if not self.ATGC:
                return 0
            return sum(list(self.ATGC.values()))
        return self.seqlength
        
    def get_pattern_skew(self, min_k: int = 0, max_k: int = 0):
        if (min_k == 0 and max_k == 0) or (min_k == self.min_k and max_k == self.max_k):
            return self._rank_distance(self.copy(reverse_complement=True), flg_pattern_skew=True)
        elif min_k < self.min_k or max_k > self.max_k:
            sys.exit(
                f"\n❌ The requested k-mer range [{min_k}..{max_k}] exceeds the available k-mer range [{self.min_k}..{self.max_k}]!"
            )
        else:
            oGenome = self.copy(min_k = min_k, max_k = max_k)
            return oGenome._rank_distance(oGenome.copy(reverse_complement=True), flg_pattern_skew=True)
            
    def get_pattern_variance(self, min_k: int = 0, max_k: int = 0):
        if (min_k == 0 and max_k == 0) or (self.min_k <= min_k <= self.max_k and self.min_k <= max_k <= self.max_k and min_k <= max_k):
            values = [float(v.split(",")[-1]) for v in self.get_values(min_k = min_k, max_k = max_k, data_type = "z-score")]
            return np.var(values)
        else:
            sys.exit(
                f"\n❌ The requested k-mer range [{min_k}..{max_k}] exceeds the available k-mer range [{self.min_k}..{self.max_k}]!"
            )
        
    def get_pattern_std(self, min_k: int = 0, max_k: int = 0):
        if (min_k == 0 and max_k == 0) or (self.min_k <= min_k <= self.max_k and self.min_k <= max_k <= self.max_k and min_k <= max_k):
            values = [float(v.split(",")[-1]) for v in self.get_values(min_k = min_k, max_k = max_k, data_type = "z-score")]
            return np.std(values)
        else:
            sys.exit(
                f"\n❌ The requested k-mer range [{min_k}..{max_k}] exceeds the available k-mer range [{self.min_k}..{self.max_k}]!"
            )
        
    def get_distance(self, other, distance: str = "hamming", data_type: str = "median_centered-z-score", flg_reverse_complement: bool = False):
        if distance.lower() == "hamming":
            return self._hamming_distance(other, flg_reverse_complement)
        elif distance.lower() == "euclidean":
            # data_type = count | z-score | median_centered-z-score
            return self._euclidean_distance(other, flg_reverse_complement, data_type=data_type)
        elif distance.lower() == "rank":
            return self._rank_distance(other, flg_reverse_complement)
        else:
            sys.exit(f"\n❌ Unrecognized distance type {distance}!")
                
    def get_gc_content(self, digits: int = 0):
        """
        Calculate the GC-content of the sequence.
    
        GC-content = (G + C) / (G + C + A + T)
    
        Parameters
        ----------
        digits : int, default=0
            If >0, return the value formatted with the specified number of
            decimal places; otherwise return a float.
    
        Returns
        -------
        float or str
            GC-content as a float (digits=0) or formatted string.
        """
        canonical_seqlength = self.get_seqlength(True)
        if canonical_seqlength == 0:
            gc_cont = 0.0
        else:    
            gc_cont = (self.ATGC["G"] + self.ATGC["C"])/canonical_seqlength
            
        if digits == 0:
            return gc_cont
        return f"{gc_cont:.{digits}f}"
        
    def get_gc_skew(self, digits: int = 0):
        """
        Calculate the GC-skew of the sequence.
    
        GC-skew = (G - C) / (G + C)
    
        Parameters
        ----------
        digits : int, default=0
            If >0, return the value formatted with the specified number of
            decimal places; otherwise return a float.
    
        Returns
        -------
        float or str
            GC-skew as a float (digits=0) or formatted string.
        """
    
        gc = self.ATGC["G"] + self.ATGC["C"]
    
        if gc == 0:
            skew = 0.0
        else:
            skew = (self.ATGC["G"] - self.ATGC["C"]) / gc
    
        if digits <= 0:
            return skew
    
        return f"{skew:.{digits}f}"
        
    def get_abs_gc_skew(self, digits: int = 0):
        return abs(self.get_gc_skew(digits=digits))
        
    def get_at_skew(self, digits: int = 0):
        """
        Calculate the AT-skew of the sequence.
    
        AT-skew = (A - T) / (A + T)
    
        Parameters
        ----------
        digits : int, default=0
            If >0, return the value formatted with the specified number of
            decimal places; otherwise return a float.
    
        Returns
        -------
        float or str
            AT-skew as a float (digits=0) or formatted string.
        """
    
        at = self.ATGC["A"] + self.ATGC["T"]
    
        if at == 0:
            skew = 0.0
        else:
            skew = (self.ATGC["A"] - self.ATGC["T"]) / at
    
        if digits <= 0:
            return skew
    
        return f"{skew:.{digits}f}"
        
    def get_abs_at_skew(self, digits: int = 0):
        return abs(self.get_at_skew(digits=digits))
        
        
    def get_purine_skew(self, digits: int = 0):
        """
        Calculate the purine-skew of the sequence.
    
        purine-skew = (G + A - C - T) / (G + C + A + T)
    
        Parameters
        ----------
        digits : int, default=0
            If >0, return the value formatted with the specified number of
            decimal places; otherwise return a float.
    
        Returns
        -------
        float or str
            purine-skew as a float (digits=0) or formatted string.
        """
        canonical_seqlength = self.get_seqlength(True)
        if canonical_seqlength == 0:
            skew = 0.0
        else:
            skew = (self.ATGC["G"] + self.ATGC["A"] - self.ATGC["C"] - self.ATGC["T"]) / canonical_seqlength
    
        if digits <= 0:
            return skew
    
        return f"{skew:.{digits}f}"
        
    def get_abs_purine_skew(self, digits: int = 0):
        return abs(self.get_purine_skew(digits-digits))
        
        
    #### PRIVATE METHODS

    def _generate_string(self):
        n_words = sum(4**k for k in range(self.min_k, self.max_k + 1))
        string = "1" + "0" * (3 * n_words)
        return int(string, 2)    
    

    def _add_word(self, word: str, count: int = 1, combine_complements: bool = False) -> None:
        wlength,x,y = self.oMapper(word)
            
        if combine_complements:
            y, x = sorted([x, y])
            
        return self.add(wlength, x, y, count)

    def _count_words(self, sequence: str, wlength: int, flg_progress: bool = True, bar_text: str = "Word count: "):
        sequence = sequence.upper()
        sorted_words = sorted([
            sequence[i:i + wlength]
            for i in range(len(sequence) - wlength + 1)
            if 'N' not in sequence[i:i + wlength]
        ])
        
        """
        Convert a sorted list of words into [[word, count], ...].
        
        Args:
            sorted_words (list[str]): Sorted list of words.
            
        Returns:
            list[list]: Each record is [word, count].
            ["ACCTG", "ACCTG", "ATG", "GCTA", "GCTA", "GCTA"] -> [['ACCTG', 2], ['ATG', 1], ['GCTA', 3]]
        """
        words = [[word, sum(1 for _ in group)] for word, group in groupby(sorted_words)]

        bar = None
        if flg_progress:
            bar = progressbar.indicator(len(words),bar_text)
        for i in range(len(words)):
            word, count = words[i]
            if sum([word.upper().count(L) for L in ["A", "T", "G", "C"]]) == len(word):
                # Add word and count to the database
                self._add_word(word,count)
                
            if bar:
                try:
                    bar(i)
                except:
                    pass
        if bar:
            bar.stop()
        return 
    
    def _set_digital_value(self, word: str, value: str):
        """
        `word` may be:
            "ATG"           # DNA word
            "3,3,5"         # k,x,y
            "ATG,3,3,5"     # word,k,x,y
            [3,3,5]         # list of indices
            ["ATG",3,3,5]   # word + indices
    
        Set the 3-bit abundance value for a k-mer.
    
        Allowed values:
            '000', '001', '011', '111'
    
        Returns True if the value was set successfully,
        or False if the k-mer is invalid.
        """
    
        allowed_values = {"000", "001", "011", "111"}
    
        if value not in allowed_values:
            raise ValueError(
                f"Invalid digital value '{value}'. "
                f"Allowed values are: {sorted(allowed_values)}"
            )
    
        # Get zero-based k-mer index
        i = self.oMapper.inline_index(word, min_k=self.min_k)
    
        if i is None:
            return False
    
        # Check that the word does not exceed max_k
        if isinstance(word, list):
            indices = word
        else:
            indices = word.split(",")
    
        if len(indices) == 4:
            k = int(indices[1])
        elif len(indices) == 3:
            k = int(indices[0])
        elif len(indices) == 1:
            k = len(word)
        else:
            return False
    
        if k > self.max_k:
            return False
    
        # Number of k-mers stored
        n_words = sum(
            4**k for k in range(self.min_k, self.max_k + 1)
        )
    
        # Position of the three bits counted from the right
        shift = 3 * (n_words - i - 1)
    
        # Clear the existing three bits
        mask = 0b111 << shift
        self.string &= ~mask
    
        # Insert the new value
        self.string |= int(value, 2) << shift
    
        return True
    
    def _deplete_sequence(self, seq: str, targer_seq_length: int, chunk_number: int) -> str:
        """
        Split a sequence into `chunk_number` chunks.
        - If len(seq) > targer_seq_length: select evenly spaced chunks whose combined length = targer_seq_length.
        - If len(seq) <= targer_seq_length: split the full sequence into `chunk_number` chunks.
    
        For both cases: reverse-complement every even-numbered chunk (2, 4, ...).
        Concatenate all chunks and return.
    
        Args:
            seq: Input nucleotide sequence (string).
            targer_seq_length: Target number of bases to keep across all chunks (if len(seq) is larger).
            chunk_number: Number of chunks to extract.
    
        Returns:
            Concatenated string of chunks (with even-numbered chunks RC'ed).
        """
        n = len(seq)
    
        if chunk_number <= 0:
            raise ValueError("chunk_number must be a positive integer.")
        if targer_seq_length < 0:
            raise ValueError("targer_seq_length must be non-negative.")
    
        # Case A: sequence is short enough → split entire sequence
        if n <= targer_seq_length:
            base_chunk_len = n // chunk_number
            rem = n % chunk_number
            chunk_lengths = [
                base_chunk_len + (1 if i < rem else 0)
                for i in range(chunk_number)
            ]
            pieces: List[str] = []
            pos = 0
            for i, clen in enumerate(chunk_lengths):
                chunk = seq[pos:pos + clen]
                if (i + 1) % 2 == 0:
                    chunk = tools.reverse_complement(chunk)
                pieces.append(chunk)
                pos += clen
            return "".join(pieces)
    
        # Case B: sequence longer than target → depletion logic
        if targer_seq_length < chunk_number:
            raise ValueError("targer_seq_length must be at least chunk_number so each chunk has ≥1 base.")
    
        # Split target length into nearly equal chunks
        base_chunk_len = targer_seq_length // chunk_number
        chunk_len_rem = targer_seq_length % chunk_number
        chunk_lengths: List[int] = [
            base_chunk_len + (1 if i < chunk_len_rem else 0)
            for i in range(chunk_number)
        ]
    
        # Compute spacers between chunks (and margins)
        total_gaps = n - targer_seq_length
        gaps_count = chunk_number + 1
        base_gap = total_gaps // gaps_count
        gap_rem = total_gaps % gaps_count
        gaps: List[int] = [
            base_gap + (1 if i < gap_rem else 0)
            for i in range(gaps_count)
        ]
    
        pieces: List[str] = []
        pos = gaps[0]
        for i, clen in enumerate(chunk_lengths):
            chunk = seq[pos:pos + clen]
            if (i + 1) % 2 == 0:
                chunk = tools.reverse_complement(chunk)
            pieces.append(chunk)
            pos += clen
            if i < chunk_number - 1:
                pos += gaps[i + 1]
    
        return "".join(pieces)

    def _transform_to_digital_value(self, wlength: int, count: int, seqlength: int, binary_code: bool = False, coding: list = []):
        thresholds = {
            2: [48840, 68125, 76837],
            3: [16122, 21025, 24081],
            4: [3165, 4747, 5986],
            5: [820, 1173, 1638],
            6: [178, 278, 410],
            7: [41.0, 66.8, 104.7],
            8: [9.1, 15.7, 26.3],
            9: [2.1, 3.85, 6.74],
            10: [0.5752, 1.0286, 1.9521],
            11: [0.28756, 0.41772, 0.683],
        }
        
        f = 1000000 * count / seqlength
        
        level = 3
        for i in range(3):
            if f <= thresholds[wlength][i]:
                level = i
                break
        if binary_code:
            values = ["000", "001", "011", "111"]
            return values[level]
        if coding and len(coding) == 4:
            return coding[level]
        return level
        
    def _reverse_complement_kmer(self, kmer: list | str):   # kmer = k,x,y or word,k,x,y
        is_string = False
        if isinstance(kmer, str):
            is_string = True
            kmer = kmer.split(",")
            
        k,x,y = [int(v) for v in kmer[-3:]]
        word = ""
        if len(kmer) == 4:
            word = self.oMapper(k,y,x)
        if is_string:
            return f"{word},{k},{y},{x}" if word else f"{k},{y},{x}"
        return [k,y,x]
            
    def _is_lower(self, kmer: list):
        k,x,y = [int(v) for v in kmer[-3:]]
        if x <= y:
            return True
        return False

    def _is_upper(self, kmer: list):
        k,x,y = [int(v) for v in kmer[-3:]]
        if x >= y:
            return True
        return False

    def _euclidean_distance(self, other, data_type: str = "median_centered-z-score", flg_reverse_complement: bool = False):     # data_type = count | z-score | median_centered-z-score
        if not isinstance(other, Genome):
            raise TypeError(
                f"Unsupported operand type '{type(other).__name__}' "
                "for genome comparison."
            )
    
        if self.min_k != other.min_k or self.max_k != other.max_k:
            raise ValueError(
                "The compared genomes use different k-mer ranges."
            )
        
        # Extract the z-score from the last comma-separated field
        # returned for each k-mer.
        a = [
            float(record.split(",")[-1])
            for record in self.get_values(data_type=data_type)
        ]
    
        b = [
            float(record.split(",")[-1])
            for record in other.get_values(data_type=data_type)
        ]
    
        # This should normally be guaranteed by the equal k-mer ranges,
        # but the check prevents silent truncation or indexing errors.
        if len(a) != len(b):
            raise ValueError(
                "The compared genomes produced z-score vectors of "
                f"different lengths: {len(a)} and {len(b)}."
            )
    
        # Euclidean distance:
        # sqrt(sum((a_i - b_i)^2))
        return np.sqrt(
            sum((value_a - value_b) ** 2 for value_a, value_b in zip(a, b))
        )
    
    # Normalized Hamming distance between digital strings
    def _hamming_distance(self, other, flg_reverse_complement: bool = False):
        if not isinstance(other, Genome):
            raise TypeError(
                f"Unsupported operand type '{type(other).__name__}' "
                "for genome comparison."
            )
    
        if self.min_k != other.min_k or self.max_k != other.max_k:
            raise ValueError(
                "The compared genomes use different k-mer ranges."
            )
    
        a = self.string
        b = other.string
    
        # Number of data bits (excluding the leading sentinel bit).
        total = a.bit_length() - 1
    
        if total <= 0:
            return 0.0
    
        # XOR has a 1 wherever the two strings differ.
        return (a ^ b).bit_count() / total
            
    # Compare k-mer z-score ranks    
    def _rank_distance(self, other, flg_reverse_complement: bool = False, flg_pattern_skew: bool = False):
        """
        Calculate the normalized rank distance between the k-mer
        z-score patterns of two genomes.
    
        The distance is accumulated across all k-mer lengths from
        self.min_k through self.max_k.
        """
    
        if not isinstance(other, Genome):
            raise TypeError(
                f"Unsupported operand type '{type(other).__name__}' "
                "for genome comparison."
            )
    
        if self.min_k != other.min_k or self.max_k != other.max_k:
            raise ValueError(
                "The compared genomes use different k-mer ranges: "
                f"[{self.min_k}..{self.max_k}] and "
                f"[{other.min_k}..{other.max_k}]."
            )
    
        distance = 0
        D_min = 0
        D_max = 0
    
        for k in range(self.min_k, self.max_k + 1):
    
            # Each record is expected to end with the z-score, for example:
            #
            #     "ATG,3,3,5,-1.247"
            #
            # Everything except the last field is retained as the unique
            # k-mer identifier.
            a = []
    
            for record in self.get_values(
                min_k=k,
                max_k=k,
                data_type="z-score",
                flg_add_kmers=True,
            ):
                parts = record.split(",")
    
                try:
                    z_score = float(parts[-1])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid z-score in record '{record}'."
                    ) from exc
    
                kmer = ",".join(parts[:-1])
                a.append((kmer, z_score))
    
            b = []
    
            for record in other.get_values(
                min_k=k,
                max_k=k,
                data_type="z-score",
                flg_add_kmers=True,
            ):
                parts = record.split(",")
    
                try:
                    z_score = float(parts[-1])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid z-score in record '{record}'."
                    ) from exc
    
                kmer = ",".join(parts[:-1])
                b.append((kmer, z_score))
    
            if len(a) != len(b):
                raise ValueError(
                    f"The compared genomes produced different numbers "
                    f"of {k}-mers: {len(a)} and {len(b)}."
                )
    
            # Sort by z-score. The k-mer identifier is used as a secondary
            # key to ensure reproducible ordering when z-scores are equal.
            a.sort(key=lambda item: (item[1], item[0]))
            b.sort(key=lambda item: (item[1], item[0]))
    
            # Assign a zero-based rank to every k-mer.
            a_ranks = {
                kmer: rank
                for rank, (kmer, _) in enumerate(a)
            }
    
            b_ranks = {
                kmer: rank
                for rank, (kmer, _) in enumerate(b)
            }
    
            # The same k-mer set must occur in both genomes.
            if a_ranks.keys() != b_ranks.keys():
                raise ValueError(
                    f"The compared genomes produced different sets "
                    f"of {k}-mers."
                )
    
            # Add the absolute rank differences for this k-mer length.
            distance += sum(
                abs(a_ranks[kmer] - b_ranks[kmer])
                for kmer in a_ranks
            )
    
            # Accumulate the theoretical distance limits.
            #
            # D_min remains zero for comparison of independent genomes.
            D_max += 4**k * (4**k - 1) / 2
            if flg_pattern_skew:
                D_min += 4**k if k % 2 == 1 else 4**k - 2**k
    
        if D_max == D_min:
            return 0.0
    
        return 100.0 * (distance - D_min) / (D_max - D_min)

###################################################
if __name__ == "__main__":
    '''
    oWDB = WordDB(min_k=2, max_k=4)
    
    for fname in [fn for fn in os.listdir("test") if fn.endswith(".gbk")]:
        input_path = os.path.join("test", fname)
        oWDB.add_genome(input_path)
        
    oWDB.save_dbfile("test.pkl")
    '''
    oWDB = WordDB()
    oWDB.open_dbfile("test.pkl")
    #print(oWDB[0].title, oWDB[0].seqlength, oWDB[0].get_pattern_variance(min_k=4, max_k=4))
    #oWDB.cluster_genomes(distance_type="euclidean", out_file="matrix.nwk")
    oWDB.get_matrix(min_k=3, max_k=3, label_taxon="genus", out_file="matrix.tsv")

    
    
