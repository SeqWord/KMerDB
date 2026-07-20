import os, sys, math, re
from typing import List, Optional, Dict, Tuple
import progressbar

# In-house modules
import tools, CSV_merger
from word_db import WordDB
from dendro import Root as tree
from spectra import spectral_analysis as spa

def execute(path: str, min_k: int = 4, max_k: int = 5,
    algorithm: str = "SPECTRAL",          # NJ | UPGMA | SPECTRAL | "" - return value matrix (v_matrix)
    output_file: str = "",
    output_format: str = "",
    max_cluster_number: int = 5, max_cluster_content: int = 5, 
    max_levels: int = 5, force_k: int = 0,
    matrix_type: str = "D", # D | V for distance or value matrix
    ) -> tuple[tree, list[str]]:
    
    # Asserten file availability
    if not os.path.exists(path):
        tools.msg(f"Project folder {path} does not exist!")
        sys.exit()
        
    wdb_files = [fn for fn in os.listdir(path) if fn.lower().endswith(".wdb")]
    if len(wdb_files) == 0:
        tools.msg(f"Project folder {path} does not contain any WDB files!")
        sys.exit()
        
    total = 0
        
    genomes = []
    matrix = []
    values = []
    v_matrix = []
    bar = progressbar.indicator(len(wdb_files),"Process input files... ")
    counter = 0
    
    for fname in wdb_files:
        # Open WDB file
        oDB = openDBFile(os.path.join(path, fname))
        if oDB == None:
            continue
        
        # Extract data from WDB object
        data = oDB.export_db(min_k, max_k - 1)
        words = data["words"]
        if not v_matrix and matrix_type == "V":
            v_matrix = [["Genome"] + [word['word'] for word in words]]
        
        # Sort genomes by ID
        data['genomes'].sort(key=lambda d: int(d['ID']))
        for g in data['genomes']:
            g['file'] = os.path.basename(fname)
        genomes += data['genomes']
        
        # Transpose 'values' from [words][genomes] to [genomes][words]
        # "values":[['001', '111', '011', ...],...],    # Number of lists = number of genomes, number of values per list = number of words
        # matrix = [[ls[int(genome['ID']) - 1] for ls in data['values']] for genome in data['genomes']]
        for i in range(len(genomes)):
            matrix.append([])
            for j in range(len(words)):
                matrix[i].append(data['values'][j][i])
        
        # Set total
        if not total:
            total = len(matrix[0]) * 3
            
        # Each value is a long integer
        values = [int("1"+"".join(ls), 2) for ls in matrix]
        
        counter += 1
        bar(counter)
    
    bar.stop()
    
    # Set value v_matrix
    if matrix_type == "V":
        mapping = {
            '000': '-2',
            '001': '-1',
            '011':  '1',
            '111':  '2'
        }
        for i in range(len(genomes)):
            v_matrix.append([genomes[i]['accession']] + [mapping[value] for value in matrix[i]])
            
        delimiter = "\t"
        if output_format.upper() == "CSV":
            delimiter = ","
        tools.saveTextFile(
            strText = "\n".join([delimiter.join(line) for line in v_matrix]),
            fname = output_file)
        return
    
    # Set distance d_matrix
    d_matrix = []
    for i in range(len(values) - 1):
        d_matrix.append([])
        for j in range(i + 1, len(values)):
            a, b = [values[i], values[j]]
            d_matrix[-1].append((a ^ b).bit_count() / total)
            
    # Collection of paths 'Root>1>1>0>Label'
    labels = [f"{genome['file']}_{genome['accession']}" for genome in genomes]
    
    if algorithm.upper() == "UPGMA":
        pathways = upgma_paths_from_upper_triangle(d_matrix, labels, output_file=output_file)
    elif algorithm.upper() == "NJ":
        pathways = nj_paths_from_upper_triangle(d_matrix, labels, output_file=output_file)
    elif algorithm.upper() == "SPECTRAL":
        pathways = spa(matrix=d_matrix, 
            labels=labels, 
            max_cluster_number=max_cluster_number, 
            max_cluster_content=max_cluster_content,
            max_levels=max_levels,
            force_k = force_k,
            output_format=output_format,    # pathways/newick
            output_file=output_file         # save output in a separate file
        )
    else:
        tools.msg(f"Unknown clustering algoritm {algorithm}!")
        sys.exit()
    
    """
    # Create oTree object
    oTree = tree(description = os.path.basename(path))
    
    #### ERROR
    for pathway in pathways:
        oTree.append(pathway)
    """
    
    return pathways

# -------------------------------
# Open a custom WordDB file
# -------------------------------
def openDBFile(path):
    oDB = WordDB()
    try:
        oDB.open_dbfile(path)
    except:
        tools.alert(f"Problem with opening {path}!", "Alert!")
        return None
    return oDB

# -------------------------------
# Calculate paths from root to end-leafes of constructed UPGMA tree
# -------------------------------
def upgma_paths_from_upper_triangle(
    tri: List[List[float]], 
    labels: Optional[List[str]] = None,
    output_file: str = "",
) -> List[str]:
    """
    Build a UPGMA tree from an upper-triangle distance list-of-lists and
    return paths from root to each leaf as 'root>0>1>...>Label'.

    If `output_file` is a non-empty path, also save the tree in
    Newick format to that file (UTF-8).

    tri[i] contains distances from taxon i to (i+1 ... n-1).
    Example for n=4:
        tri = [
            [d01, d02, d03],
            [     d12, d13],
            [          d23],
        ]
    labels: optional list of n leaf names; defaults to ['0','1',...].
    """
    n = len(tri) + 1
    if labels is None:
        labels = [str(i) for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match number of taxa")

    # Expand to full symmetric d_matrix
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n-1):
        row = tri[i]
        if len(row) != n-1-i:
            raise ValueError(f"Row {i} should have {n-1-i} items")
        for k, val in enumerate(row, start=1):
            j = i + k
            d = float(val)
            dist[i][j] = d
            dist[j][i] = d

    # Nodes:
    #  - leaves: {'size', 'height', 'label'}
    #  - internals: {'size','height','children':(L,R)}
    nodes: Dict[int, Dict] = {i: {'size': 1, 'height': 0.0, 'label': labels[i]} for i in range(n)}
    active = list(range(n))
    D = {i: {j: dist[i][j] for j in range(n)} for i in range(n)}
    next_id = n

    while len(active) > 1:
        # Find closest pair (stable tie-break by IDs)
        best = (math.inf, None, None)
        for ai in range(len(active)):
            a = active[ai]
            for bi in range(ai+1, len(active)):
                b = active[bi]
                d = D[a][b]
                key = (d, min(a, b), max(a, b))
                if key < best:
                    best = key
        best_dist, a, b = best
        if a is None:
            raise RuntimeError("No pair found")

        # New cluster height and children (left=0 -> 'a', right=1 -> 'b')
        hc = best_dist / 2.0
        size_c = nodes[a]['size'] + nodes[b]['size']
        c = next_id; next_id += 1
        nodes[c] = {'size': size_c, 'height': hc, 'children': (a, b)}

        # Update distances to new cluster: size-weighted average
        D[c] = {}
        for k in active:
            if k in (a, b):
                continue
            dak, dbk = D[a][k], D[b][k]
            D[c][k] = D[k][c] = (nodes[a]['size'] * dak + nodes[b]['size'] * dbk) / size_c

        # Remove a,b; clean maps
        active = [x for x in active if x not in (a, b)]
        for x in list(D.keys()):
            if x in (a, b):
                D.pop(x, None)
            else:
                D[x].pop(a, None); D[x].pop(b, None)

        # Add new cluster
        active.append(c)

    root = active[0]

    # DFS to collect paths (root>0/1>...>Label)
    paths: List[str] = []
    def dfs(node_id: int, prefix: str):
        node = nodes[node_id]
        if 'label' in node:  # leaf
            paths.append(f"{prefix}>{node['label']}")
        else:
            left, right = node['children']
            dfs(left,  f"{prefix}>0")
            dfs(right, f"{prefix}>1")

    dfs(root, "root")

    # --- Also produce a Newick string and save if requested ---
    # In UPGMA (ultrametric), branch length from node -> parent = parent.height - node.height
    def _safe_label_for_newick(s: str) -> str:
        # Allow simple labels unquoted; otherwise quote and escape single quotes
        if re.match(r"^[A-Za-z0-9_.|\-]+$", s):
            return s
        return "'" + s.replace("'", "''") + "'"

    def _build_newick(node_id: int, parent_height: Optional[float]) -> str:
        node = nodes[node_id]
        # Leaves carry their label; branch length is to parent (if any)
        if 'label' in node:
            if parent_height is None:
                # Single-leaf degenerate case
                return f"{_safe_label_for_newick(node['label'])}"
            bl = max(0.0, parent_height - node['height'])
            return f"{_safe_label_for_newick(node['label'])}:{bl:.6f}"
        # Internal node: build children at this node's height
        left, right = node['children']
        left_s  = _build_newick(left,  node['height'])
        right_s = _build_newick(right, node['height'])
        if parent_height is None:
            # Root: no branch length after root
            return f"({left_s},{right_s})"
        bl = max(0.0, parent_height - node['height'])
        return f"({left_s},{right_s}):{bl:.6f}"

    newick = _build_newick(root, None) + ";"
    if output_file:
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(newick + "\n")

    return paths

# -------------------------------
# Calculate paths from root to end-leafes of constructed NJ tree
# -------------------------------
def nj_paths_from_upper_triangle(
    tri: List[List[float]],
    labels: Optional[List[str]] = None,
    decimals: int = 4,
    clamp_negative_to_zero: bool = True,
    output_file: str = "",
) -> List[str]:
    """
    Neighbor-Joining (NJ) from an upper-triangle distance list-of-lists.
    Returns paths like: root>(len)0>(len)1>(len)Label (left=0, right=1).
    Optionally also saves the resulting tree in Newick format if
    `output_file` is a non-empty path.

    Parameters
    ----------
    tri : list of lists
        tri[i] contains distances from taxon i to taxa (i+1 ... n-1).
        Example for n=4:
            tri = [
                [d01, d02, d03],
                [     d12, d13],
                [          d23],
            ]
    labels : list of str, optional
        Names for leaves (default: "0","1",...).
    decimals : int
        Digits after decimal for branch lengths.
    clamp_negative_to_zero : bool
        If True, negative branch lengths (possible with noisy data) are set to 0.
    output_file : str
        If non-empty, write the Newick string to this file (UTF-8).
    """
    n = len(tri) + 1
    if labels is None:
        labels = [str(i) for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match number of taxa")

    # Expand to full symmetric d_matrix
    D: Dict[int, Dict[int, float]] = {i: {j: (0.0 if i == j else 0.0) for j in range(n)} for i in range(n)}
    for i in range(n - 1):
        row = tri[i]
        if len(row) != n - 1 - i:
            raise ValueError(f"Row {i} should have {n-1-i} items")
        for k, val in enumerate(row, start=1):
            j = i + k
            d = float(val)
            D[i][j] = d
            D[j][i] = d

    # Node store: leaves have {'label'}, internals have {'children': [(child_id, bl), (child_id, bl)]}
    nodes: Dict[int, Dict] = {i: {"label": labels[i]} for i in range(n)}
    active: List[int] = list(range(n))
    next_id = n

    def fmt(x: float) -> str:
        x = max(0.0, x) if clamp_negative_to_zero else x
        return f"{x:.{decimals}f}"

    while len(active) > 2:
        m = len(active)
        # r_i = sum of distances from i to all others in active
        r = {i: sum(D[i][k] for k in active if k != i) for i in active}

        # Q-d_matrix: Q_ij = (m-2)*D_ij - r_i - r_j
        best = (math.inf, None, None)
        for a_idx in range(len(active)):
            i = active[a_idx]
            for b_idx in range(a_idx + 1, len(active)):
                j = active[b_idx]
                Q = (m - 2) * D[i][j] - r[i] - r[j]
                key = (Q, min(i, j), max(i, j))  # deterministic tie-break
                if key < best:
                    best = key
        _, i, j = best
        if i is None:
            raise RuntimeError("Failed to find a pair to join")

        # Branch lengths from new node u to i and j
        dij = D[i][j]
        li = 0.5 * dij + (r[i] - r[j]) / (2 * (m - 2))
        lj = dij - li
        if clamp_negative_to_zero:
            li = max(0.0, li)
            lj = max(0.0, lj)

        # Create new node u with ordered children (left=smaller id)
        left, right = (i, j) if i < j else (j, i)
        bl_left, bl_right = (li, lj) if left == i else (lj, li)
        u = next_id
        next_id += 1
        nodes[u] = {"children": [(left, bl_left), (right, bl_right)]}

        # Distances from u to others: D_u,k = 0.5*(D_i,k + D_j,k - D_i,j)
        D[u] = {}
        for k in active:
            if k in (i, j):
                continue
            Duk = 0.5 * (D[i][k] + D[j][k] - dij)
            D[u][k] = Duk
            D[k][u] = Duk
        D[u][u] = 0.0

        # Remove i and j from D and active
        for x in (i, j):
            for k in list(D.keys()):
                if x in D.get(k, {}):
                    D[k].pop(x, None)
            D.pop(x, None)
        active = [x for x in active if x not in (i, j)]
        active.append(u)

    # Final join: two active nodes become children of root
    a, b = sorted(active)
    root = next_id
    next_id += 1
    lab = D[a][b] / 2.0
    rab = D[a][b] - lab
    if clamp_negative_to_zero:
        lab = max(0.0, lab)
        rab = max(0.0, rab)
    nodes[root] = {"children": [(a, lab), (b, rab)]}

    # --- Build paths for the original return value ---
    paths: List[str] = []

    def is_leaf(node_id: int) -> bool:
        return "label" in nodes[node_id]

    def dfs(parent_id: int, prefix: str):
        left, right = nodes[parent_id]["children"]
        for idx, (child, bl) in enumerate((left, right)):
            length_seg = f">({fmt(bl)})"
            if is_leaf(child):
                paths.append(prefix + length_seg + nodes[child]["label"])
            else:
                bit = "0" if idx == 0 else "1"
                dfs(child, prefix + length_seg + bit)

    dfs(root, "root")

    # --- Also produce a Newick string and save if requested ---
    def _safe_label_for_newick(s: str) -> str:
        # Quote if label has characters outside the simple set; escape single quotes by doubling.
        if re.match(r"^[A-Za-z0-9_.|\-]+$", s):
            return s
        return "'" + s.replace("'", "''") + "'"

    def _subtree(child_id: int, bl: float) -> str:
        if is_leaf(child_id):
            return f"{_safe_label_for_newick(nodes[child_id]['label'])}:{fmt(bl)}"
        (l_id, l_bl), (r_id, r_bl) = nodes[child_id]["children"]
        return f"({_subtree(l_id, l_bl)},{_subtree(r_id, r_bl)}):{fmt(bl)}"

    # Root has two children but no own branch length; standard Newick: (subtreeA,subtreeB);
    (l_id, l_bl), (r_id, r_bl) = nodes[root]["children"]
    newick = f"({_subtree(l_id, l_bl)},{_subtree(r_id, r_bl)});"

    if output_file:
        print("make_tree:427", output_file)
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(newick + "\n")

    return paths
