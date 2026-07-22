from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Sequence, Dict, Union
import networkx as nx
import math, random
import numpy as np
import tools

# --------------------------- Type aliases ---------------------------
Number = Union[int, float]
TriList = List[List[Number]]
MatrixLike = Union[TriList, List[List[Number]], np.ndarray]

# --------------------------- Utilities ---------------------------

def _expand_lower_triangle(lower: TriList) -> np.ndarray:
    """
    lower[i] holds distances from taxon i+1 down to 0 (length i).
    Example for n=4:
        lower = [
            [d10],
            [d20, d21],
            [d30, d31, d32],
        ]
    """
    n = len(lower) + 1
    D = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        row = lower[i - 1]
        if len(row) != i:
            raise ValueError(f"Lower-tri row {i-1} length {len(row)} != expected {i}")
        for j in range(i):
            val = float(row[j])
            D[i, j] = val
            D[j, i] = val
    return D

def _expand_upper_triangle(upper: TriList) -> np.ndarray:
    """
    upper[i] holds distances from taxon i to (i+1 ... n-1).
    Example for n=4:
        upper = [
            [d01, d02, d03],
            [     d12, d13],
            [          d23],
        ]
    """
    n = len(upper) + 1
    D = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        row = upper[i]
        expected = n - i - 1
        if len(row) != expected:
            raise ValueError(f"Upper-tri row {i} length {len(row)} != expected {expected}")
        for k, val in enumerate(row, start=i + 1):
            v = float(val)
            D[i, k] = v
            D[k, i] = v
    return D

def _validate_full_square_symmetric(M: Sequence[Sequence[Number]], atol: float = 1e-12) -> np.ndarray:
    """Validate a full matrix is square and (near-)symmetric; return as float ndarray."""
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Full matrix must be square; got shape {arr.shape}")
    if not np.allclose(arr, arr.T, atol=atol):
        raise ValueError("Full matrix must be symmetric (within tolerance).")
    return arr

def _expand_distance_any(matrix: MatrixLike) -> np.ndarray:
    """
    Accepts lower-triangle, upper-triangle, or full matrix and returns a full symmetric ndarray.
    Heuristic:
        - len(first row) < len(last row)  -> lower triangle
        - len(first row) > len(last row)  -> upper triangle
        - len(first row) == len(last row) -> full matrix
    """
    if isinstance(matrix, np.ndarray):
        return _validate_full_square_symmetric(matrix)

    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix must be a non-empty list (or numpy array).")

    # 1x1 special-case
    if isinstance(matrix[0], (int, float)) or (isinstance(matrix[0], list) and len(matrix) == 1 and len(matrix[0]) == 1):
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        return _validate_full_square_symmetric(arr)

    first_len = len(matrix[0])
    last_len = len(matrix[-1])

    if first_len < last_len:
        return _expand_lower_triangle(matrix)  # type: ignore[arg-type]
    elif first_len > last_len:
        return _expand_upper_triangle(matrix)  # type: ignore[arg-type]
    else:
        return _validate_full_square_symmetric(matrix)

def _median_nonzero_upper(a: np.ndarray) -> float:
    vals = a[np.triu_indices_from(a, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))

def _gaussian_similarity(dist: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    if sigma is None:
        sigma = _median_nonzero_upper(dist)
        if sigma <= 0:
            sigma = 1.0
    W = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)  # standard for spectral clustering
    return W

def _normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    eps = 1e-12
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = np.diag(inv_sqrt)
    return np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

def _eigengap_k(evals: np.ndarray, k_min: int = 2, k_max: Optional[int] = None) -> int:
    n = len(evals)
    if k_max is None:
        k_max = max(2, n // 2)
    k_max = min(k_max, n - 1)
    k_min = max(k_min, 2)
    vals = np.sort(evals)
    gaps = vals[1:k_max+1] - vals[0:k_max]
    idx = int(np.argmax(gaps)) + 1
    return max(k_min, min(k_max, idx))

def _kmeans_pp_init(X: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    n, d = X.shape
    centers = np.empty((k, d), dtype=float)
    i0 = rng.randrange(n)
    centers[0] = X[i0]
    dist2 = np.full(n, np.inf)
    for c in range(1, k):
        d2_new = np.sum((X - centers[c-1])**2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
        s = float(dist2.sum())
        if not np.isfinite(s) or s <= 0:
            centers[c] = X[rng.randrange(n)]
            continue
        probs = dist2 / s
        r = rng.random()
        cum = 0.0
        chosen = 0
        for i in range(n):
            cum += probs[i]
            if r <= cum:
                chosen = i
                break
        centers[c] = X[chosen]
    return centers

def _kmeans(X: np.ndarray, k: int, max_iter: int = 200, n_init: int = 5, seed: int = 13) -> np.ndarray:
    rng = random.Random(seed)
    best_labels = None
    best_inertia = math.inf
    for _ in range(n_init):
        centers = _kmeans_pp_init(X, k, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(max_iter):
            d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                centers[c] = X[mask].mean(axis=0) if np.any(mask) else X[rng.randrange(X.shape[0])]
        inertia = float(np.sum((X - centers[labels])**2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels

def _spectral_cluster_from_distance(dist: np.ndarray,
                                    force_k: Optional[int] = None) -> List[List[int]]:
    W = _gaussian_similarity(dist)
    Lsym = _normalized_laplacian(W)
    evals, evecs = np.linalg.eigh(Lsym)  # ascending
    if force_k is None:
        k = _eigengap_k(evals, k_min=2, k_max=None)
        k = max(2, min(k, dist.shape[0]))
    else:
        k = max(2, min(force_k, dist.shape[0]))
    U = evecs[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y = U / norms
    labels = _kmeans(Y, k=k)
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    return list(clusters.values())

# --------- Average inter-cluster distance (computed from ORIGINAL distances) ---------

def _avg_distance_between_groups(group_a: Sequence[int],
                                 group_b: Sequence[int],
                                 full_dist: np.ndarray) -> float:
    a = np.array(group_a, dtype=int)
    b = np.array(group_b, dtype=int)
    d = full_dist[np.ix_(a, b)]
    return float(d.mean()) if d.size else 0.0

def _reduce_distance_matrix_by_average(clusters: List[List[int]],
                                       full_dist: np.ndarray) -> np.ndarray:
    m = len(clusters)
    R = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            Rij = _avg_distance_between_groups(clusters[i], clusters[j], full_dist)
            R[i, j] = R[j, i] = Rij
    return R

# --------- Split clusters that exceed max_cluster_content using submatrix spectral ---------

def _split_large_once(nested: List[List[object]],
                      idx_level: List[List[int]],
                      D_full: np.ndarray,
                      labels: List[str],
                      max_cluster_content: int,
                      force_k: Optional[int]) -> Tuple[bool, List[List[object]], List[List[int]]]:
    """
    Split any cluster whose LEAF COUNT exceeds max_cluster_content by reclustering
    its submatrix (from ORIGINAL distances). Returns (changed, new_nested, new_idx_level).

    Invariant:
      - idx_level: lists of GLOBAL indices (ints)
      - nested:    parallel lists of LABEL STRINGS (presentation only)
    """
    changed = False
    new_nested: List[List[object]] = []
    new_idx_level: List[List[int]] = []

    for node, leaves in zip(nested, idx_level):
        if len(leaves) <= max_cluster_content:
            new_nested.append(node)
            new_idx_level.append(leaves)
            continue

        # Re-cluster this big cluster using submatrix
        subD = D_full[np.ix_(leaves, leaves)]
        sub_groups = _spectral_cluster_from_distance(subD, force_k=force_k)

        # If no effective split happened, try forcing k=2 once
        if len(sub_groups) <= 1 and force_k is None and len(leaves) >= 2:
            sub_groups = _spectral_cluster_from_distance(subD, force_k=2)

        if len(sub_groups) <= 1:
            # cannot split; keep as is
            new_nested.append(node)
            new_idx_level.append(leaves)
            continue

        # Build children using GLOBAL indices -> labels
        child_indices: List[List[int]] = [[leaves[i] for i in g] for g in sub_groups]
        child_nodes:   List[List[str]] = [[labels[idx] for idx in grp] for grp in child_indices]

        # Insert children as separate top-level nodes (expansion)
        new_nested.extend(child_nodes)
        new_idx_level.extend(child_indices)
        changed = True

    return changed, new_nested, new_idx_level

def nested_list_to_pathways(cl: list, root: str = "Root") -> List[str]:
    """
    Convert a multilevel nested list (internal nodes as lists/tuples, leaves as labels)
    into a list of path strings like 'Root>0>1>Leaf'.

    Examples:
      - ["A","B","C"]
          -> ['Root>A', 'Root>B', 'Root>C']              # flat top-level => no indices at root
      - [["A","B"], ["C","D"]]
          -> ['Root>0>A', 'Root>0>B', 'Root>1>C', 'Root>1>D']
      - [[["A","B"], "C"], ["D"]]
          -> ['Root>0>0>A', 'Root>0>0>B', 'Root>0>1>C', 'Root>1>D']

    Rules:
      - Any non-list/tuple value is a leaf (converted to str).
      - Lists/tuples are internal nodes; child order defines indices 0..k-1.
      - If the top level is flat (all leaves), omit indices at the root level.
    """
    def _is_list_like(x: Any) -> bool:
        return isinstance(x, (list, tuple))

    # Special-case: top-level flat => no indices at root
    top_is_flat = _is_list_like(cl) and all(not _is_list_like(ch) for ch in cl)

    paths: List[str] = []

    def _walk(node: Any, prefix: str, at_root: bool) -> None:
        if _is_list_like(node):
            for idx, child in enumerate(node):
                if _is_list_like(child):
                    # descend into sublist; include index unless flat root
                    next_prefix = prefix if (at_root and top_is_flat) else f"{prefix}>{idx}"
                    _walk(child, next_prefix, at_root=False)
                else:
                    # leaf; include index unless flat root
                    leaf = str(child)
                    if at_root and top_is_flat:
                        paths.append(f"{prefix}>{leaf}")
                    else:
                        paths.append(f"{prefix}>{idx}>{leaf}")
        else:
            # Degenerate case: tree is a single leaf
            paths.append(f"{prefix}>{str(node)}")

    _walk(cl, root, at_root=True)
    return paths
    
    
def nested_list_to_newick(cl: list) -> str:
    """
    Convert a multilevel nested list (internal nodes as lists, leaves as labels)
    into a Newick tree string without branch lengths.

    Examples of accepted shapes:
      - ["A","B","C"]                     -> (A,B,C);
      - [["A","B"], ["C","D"]]            -> ((A,B),(C,D));
      - [[["A","B"], "C"], ["D"]]         -> (((A,B),C),(D));

    Rules:
      - Any non-list value is treated as a leaf label (converted to str).
      - Lists are treated as internal nodes whose children are their elements.
      - Labels that contain spaces or Newick-special chars are single-quoted,
        and internal single quotes are doubled (Newick escaping).
    """
    def _is_list_like(x: Any) -> bool:
        return isinstance(x, (list, tuple))

    def _escape_label(x: Any) -> str:
        s = str(x)
        # Newick needs quoting if label has whitespace or any of ,():;[]
        if any(c in s for c in " \t\r\n,:;()[]"):
            s = "'" + s.replace("'", "''") + "'"
        return s

    def _to_newick(node: Any) -> str:
        if _is_list_like(node):
            # Filter out empty lists defensively
            children = [ _to_newick(ch) for ch in node if not (_is_list_like(ch) and len(ch) == 0) ]
            if not children:
                # Degenerate empty node -> anonymous leaf
                return "''"
            return "(" + ",".join(children) + ")"
        else:
            return _escape_label(node)

    out = _to_newick(cl)
    # If the entire tree is just a single label (no parens), wrap it
    if not (out.startswith("(") and out.endswith(")")):
        out = f"({out})"
    return out + ";"
    
# ----------------- Main: hierarchical spectral analysis -----------------

def spectral_analysis(matrix: MatrixLike,
                      labels: List[str],
                      max_cluster_number: int = 5,
                      max_cluster_content: int = 5,
                      output_file: str = "",          # save output in a separate file
                      output_format: str = "",        # 'newick'/'pathways'
                      force_k: Optional[int] = None,
                      max_levels: int = 100) -> Union[List[List[object]], str]:
    """
    Hierarchical spectral clustering with:
      (A) top-level reduction if number of clusters > max_cluster_number
      (B) within-cluster splitting if any cluster contains > max_cluster_content leaves

    Returns a nested list where leaves are label strings and internal nodes are lists.
    """
    # Expand to a full matrix first, then validate labels
    D_full = _expand_distance_any(matrix)
    n = D_full.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) must equal number of taxa ({n})")

    # Level-1 spectral clustering on leaves
    first_groups_idx = _spectral_cluster_from_distance(D_full, force_k=force_k)
    nested: List[List[object]] = [[labels[i] for i in grp] for grp in first_groups_idx]
    idx_level: List[List[int]]   = [grp[:] for grp in first_groups_idx]

    # Iterate hierarchy construction
    levels_done = 0
    while levels_done < max_levels:
        changed_any = False

        # (B) Split oversized clusters by reclustering submatrix until none remain
        while True:
            changed, nested, idx_level = _split_large_once(
                nested, idx_level, D_full, labels, max_cluster_content, force_k
            )
            changed_any = changed_any or changed
            if not changed:
                break

        # (A) If too many top-level clusters, reduce by clustering cluster-centroids via average distances
        if len(idx_level) > max_cluster_number:
            R = _reduce_distance_matrix_by_average(idx_level, D_full)
            higher_groups_idx = _spectral_cluster_from_distance(R, force_k=force_k)

            # Wrap: new parents grouping current top-level nodes
            new_nested: List[List[object]] = []
            new_idx_level: List[List[int]] = []
            for group in higher_groups_idx:
                parent_node: List[object] = []
                parent_leaves: List[int] = []
                for g in group:
                    parent_node.append(nested[g])      # keep entire subtree
                    parent_leaves.extend(idx_level[g]) # accumulate leaf indices
                new_nested.append(parent_node)
                # keep unique order (stable) for leaves inside the parent
                seen = set()
                ordered = []
                for x in parent_leaves:
                    if x not in seen:
                        seen.add(x)
                        ordered.append(x)
                new_idx_level.append(ordered)
            nested, idx_level = new_nested, new_idx_level
            changed_any = True

        # If both constraints satisfied and no changes, stop
        if (len(idx_level) <= max_cluster_number) and not changed_any:
            break

        levels_done += 1
        # Safety: if nothing changed at this level, stop to avoid infinite loop
        if not changed_any:
            break
    
    # Processing outputs
    out = nested                                # returns List[List[object]]
    
    if output_format.lower() == "newick":
        out = nested_list_to_newick(nested)     # returns str
    elif output_format.lower() == "pathways":
        out = nested_list_to_pathways(nested)   # returns List[List[str]]
    if output_file:
        out_text = str(out)
        if output_format.lower() == "pathways":
            out_text = "\n".join([("   " * (len(line.split(">")) - 3)) + line for line in out])
        tools.saveTextFile(out_text, output_file)
    return out
    
if __name__ == "__main__":
    # Example: 5 items, lower-triangle input (no diagonal in rows)
    matrix = [
        [2.0],
        [3.0, 4.0],
        [2.0, 1.0, 5.0],
        [7.0, 6.0, 6.5, 6.8],
    ]
    labels = ["A", "B", "C", "D", "E"]

    clusters = spectral_analysis(matrix, labels, output_file="example.txt", output_format="pathways")
    print(clusters)   # e.g. [['A','B','C'], ['D','E']] or similar
