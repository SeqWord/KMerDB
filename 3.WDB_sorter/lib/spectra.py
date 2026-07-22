from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Sequence, Dict, Union
import math, random
import numpy as np
import tools
# import networkx as nx  # (unused; keep or remove)

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

# ---------- helpers for recursive hierarchy with constraints ----------

def _cluster_groups_by_average(child_sets: List[List[int]],
                               D_full: np.ndarray,
                               k: int,
                               force_k: Optional[int]) -> List[List[int]]:
    """
    Given child_sets = list of leaf-index lists (children of the current node),
    reduce them into <= k groups by clustering the average-distance matrix
    between child_sets. Returns a new list of merged child_sets.
    """
    m = len(child_sets)
    if m <= k:
        return child_sets

    # Build average-distance matrix between child sets
    R = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            dij = _avg_distance_between_groups(child_sets[i], child_sets[j], D_full)
            R[i, j] = R[j, i] = dij

    # Force the number of clusters to exactly k to respect the cap
    groups_idx = _spectral_cluster_from_distance(R, force_k=min(k, m) if force_k is None else min(k, m, force_k))

    # Merge child sets within each cluster
    merged: List[List[int]] = []
    for g in groups_idx:
        merged_leaves: List[int] = []
        seen = set()
        for ci in g:
            for u in child_sets[ci]:
                if u not in seen:
                    seen.add(u)
                    merged_leaves.append(u)
        merged.append(merged_leaves)
    return merged

def _fallback_split_by_size(leaves: List[int],
                            D_full: np.ndarray,
                            max_cluster_content: int) -> List[List[int]]:
    """
    Deterministic fallback to enforce max_cluster_content strictly when spectral
    fails to split (returns one group). We create ceil(n / max_content) groups
    using farthest-point seeding and nearest-seed assignment, then ensure each
    group size <= max_cluster_content (split if needed).
    """
    n = len(leaves)
    if n <= max_cluster_content:
        return [list(leaves)]

    k = math.ceil(n / max_cluster_content)
    subD = D_full[np.ix_(leaves, leaves)]

    # 1) Farthest-point seeding
    seeds = [0]
    seeds.append(int(np.argmax(subD[seeds[0]])))
    while len(seeds) < k:
        min_to_seeds = np.min(subD[:, seeds], axis=1)
        next_seed = int(np.argmax(min_to_seeds))
        if next_seed in seeds:
            # fallback in degeneracy
            cand = [i for i in range(n) if i not in seeds]
            next_seed = cand[0]
        seeds.append(next_seed)

    # 2) Assign each point to nearest seed
    groups_local: List[List[int]] = [[] for _ in range(k)]
    for i in range(n):
        dists = subD[i, seeds]
        j = int(np.argmin(dists))
        groups_local[j].append(i)

    # 3) Convert local -> global, enforce hard cap by chunking if necessary
    out: List[List[int]] = []
    for g in groups_local:
        if not g:
            continue
        g_global = [leaves[i] for i in g]
        for s in range(0, len(g_global), max_cluster_content):
            out.append(g_global[s:s+max_cluster_content])
    return out

def _hier_build(leaves: List[int],
                D_full: np.ndarray,
                max_cluster_number: int,
                max_cluster_content: int,
                force_k: Optional[int],
                flg_strict_split: bool) -> List[object]:
    """
    Build hierarchy for this node as follows:
      1) Always attempt ONE spectral split of `leaves`.
      2) If split yields only 1 group:
           - if len(leaves) <= max_cluster_content -> terminal (return leaf list)
           - else:
               * if flg_strict_split: force-split by fallback + cap children
               * else: allow oversized terminal (return leaf list)
      3) If split yields multiple children:
           - if children > max_cluster_number -> merge children by average distance to cap
           - recurse only for child having size > max_cluster_content
    """
    # 1) Try one spectral split on this set
    subD = D_full[np.ix_(leaves, leaves)]
    sub_groups = _spectral_cluster_from_distance(subD, force_k=force_k)

    # Fallback to k=2 if no effective split
    if len(sub_groups) <= 1 and len(leaves) >= 2:
        sub_groups = _spectral_cluster_from_distance(subD, force_k=2)

    # 2) Still one group
    if len(sub_groups) <= 1:
        if len(leaves) <= max_cluster_content:
            return list(leaves)  # terminal within cap
        # len(leaves) > cap
        if not flg_strict_split:
            return list(leaves)  # allow oversized terminal
        # strict: force-split deterministically, then cap children count
        child_sets = _fallback_split_by_size(leaves, D_full, max_cluster_content)
        if len(child_sets) > max_cluster_number:
            child_sets = _cluster_groups_by_average(child_sets, D_full,
                                                    k=max_cluster_number, force_k=force_k)
        children: List[object] = []
        for child_leaves in child_sets:
            if len(child_leaves) > max_cluster_content:
                children.append(_hier_build(child_leaves, D_full,
                                            max_cluster_number, max_cluster_content, force_k,
                                            flg_strict_split))
            else:
                children.append(list(child_leaves))
        return children

    # Map local -> global indices
    child_sets: List[List[int]] = [[leaves[i] for i in g] for g in sub_groups]

    # 3) Enforce per-node max children
    if len(child_sets) > max_cluster_number:
        child_sets = _cluster_groups_by_average(
            child_sets, D_full, k=max_cluster_number, force_k=force_k
        )

    # Recurse only for big children; small ones become terminal
    children: List[object] = []
    for child_leaves in child_sets:
        if len(child_leaves) > max_cluster_content:
            children.append(_hier_build(child_leaves, D_full,
                                        max_cluster_number, max_cluster_content, force_k,
                                        flg_strict_split))
        else:
            children.append(list(child_leaves))  # terminal cluster (leaf indices)
    return children

def _map_indices_to_labels(node: object, labels: List[str]) -> object:
    """
    Convert a nested list whose leaves are ints (global indices) into
    the same nested shape but with label strings at the leaves.
    """
    if isinstance(node, list):
        if node and all(isinstance(x, int) for x in node):
            return [labels[int(x)] for x in node]
        return [_map_indices_to_labels(ch, labels) for ch in node]
    if isinstance(node, int):
        return labels[int(node)]
    return node

def _reduce_distance_matrix_by_average(clusters: List[List[int]],
                                       full_dist: np.ndarray) -> np.ndarray:
    m = len(clusters)
    R = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            Rij = _avg_distance_between_groups(clusters[i], clusters[j], full_dist)
            R[i, j] = R[j, i] = Rij
    return R

# --------- (Legacy) Split-oversized clusters loop (not used by spectral_analysis now) ---------

def _split_large_once(nested: List[List[object]],
                      idx_level: List[List[int]],
                      D_full: np.ndarray,
                      labels: List[str],
                      max_cluster_content: int,
                      force_k: Optional[int]) -> Tuple[bool, List[List[object]], List[List[int]]]:
    """
    Retained for backward compatibility; not used by spectral_analysis below.
    """
    changed = False
    new_nested: List[List[object]] = []
    new_idx_level: List[List[int]] = []

    for node, leaves in zip(nested, idx_level):
        if len(leaves) <= max_cluster_content:
            new_nested.append(node)
            new_idx_level.append(leaves)
            continue

        subD = D_full[np.ix_(leaves, leaves)]
        sub_groups = _spectral_cluster_from_distance(subD, force_k=force_k)

        if len(sub_groups) <= 1 and force_k is None and len(leaves) >= 2:
            sub_groups = _spectral_cluster_from_distance(subD, force_k=2)

        if len(sub_groups) <= 1:
            new_nested.append(node)
            new_idx_level.append(leaves)
            continue

        child_indices: List[List[int]] = [[leaves[i] for i in g] for g in sub_groups]
        child_nodes:   List[List[str]] = [[labels[idx] for idx in grp] for grp in child_indices]

        new_nested.extend(child_nodes)
        new_idx_level.extend(child_indices)
        changed = True

    return changed, new_nested, new_idx_level

# ---------------- Paths / Newick renderers ----------------

def nested_list_to_pathways(cl: list, root: str = "Root") -> List[str]:
    """
    Convert a multilevel nested list (internal nodes as lists/tuples, leaves as labels)
    into a list of path strings like 'Root>0>1>Leaf'.
    """
    def _is_list_like(x: Any) -> bool:
        return isinstance(x, (list, tuple))

    top_is_flat = _is_list_like(cl) and all(not _is_list_like(ch) for ch in cl)
    paths: List[str] = []

    def _walk(node: Any, prefix: str, at_root: bool) -> None:
        if _is_list_like(node):
            for idx, child in enumerate(node):
                if _is_list_like(child):
                    next_prefix = prefix if (at_root and top_is_flat) else f"{prefix}>{idx}"
                    _walk(child, next_prefix, at_root=False)
                else:
                    leaf = str(child)
                    if at_root and top_is_flat:
                        paths.append(f"{prefix}>{leaf}")
                    else:
                        paths.append(f"{prefix}>{idx}>{leaf}")
        else:
            paths.append(f"{prefix}>{str(node)}")

    _walk(cl, root, at_root=True)
    return paths

def nested_list_to_newick(cl: list) -> str:
    """Convert nested list to a Newick tree string without branch lengths."""
    def _is_list_like(x: Any) -> bool:
        return isinstance(x, (list, tuple))
    def _escape_label(x: Any) -> str:
        s = str(x)
        if any(c in s for c in " \t\r\n,:;()[]"):
            s = "'" + s.replace("'", "''") + "'"
        return s
    def _to_newick(node: Any) -> str:
        if _is_list_like(node):
            children = [_to_newick(ch) for ch in node if not (_is_list_like(ch) and len(ch) == 0)]
            if not children:
                return "''"
            return "(" + ",".join(children) + ")"
        else:
            return _escape_label(node)
    out = _to_newick(cl)
    if not (out.startswith("(") and out.endswith(")")):
        out = f"({out})"
    return out + ";"

# ----------------- Main: hierarchical spectral analysis -----------------

def spectral_analysis(                             
          matrix: MatrixLike,          # 'matrix': distance matrix (can be lower-triangular, upper-triangular, or full)
          labels: List[str],           # 'labels': list of names/IDs for the taxa or objects corresponding to rows/columns in matrix
          max_cluster_number: int = 10, # 'max_cluster_number': maximum number of child clusters allowed at each internal node
          max_cluster_content: int = 10,# 'max_cluster_content': maximum number of leaves allowed in a terminal cluster
          output_file: str = "",       # 'output_file': optional filename; if provided, results will be saved to this file
          output_format: str = "",     # 'output_format': format of output; can be 'newick', 'pathways', or left empty for nested list
          force_k: Optional[int] = None,# 'force_k': if set, forces spectral clustering to use exactly this number of clusters
          max_levels: int = 100,       # 'max_levels': safety limit to prevent infinite recursion (max depth of hierarchy)
          flg_strict_split: bool = False # 'flg_strict_split': if True, strictly enforce max_cluster_content using fallback split
      ) -> Union[List[object], List[str], str]:  
          # Function returns either:
          # - a nested list structure of clusters (List[object])
          # - a list of pathway strings (List[str]) if output_format='pathways'
          # - a Newick tree string (str) if output_format='newick'
    """
    Hierarchical spectral clustering with per-node constraints:
      - every internal node has at most `max_cluster_number` children
      - terminal cluster size rule depends on `flg_strict_split`:
          * False (default): allow oversized terminal if spectral can't split
          * True: enforce max_cluster_content by deterministic fallback split
    """
    D_full = _expand_distance_any(matrix)
    n = D_full.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) must equal number of taxa ({n})")

    all_leaves = list(range(n))
    idx_tree = _hier_build(all_leaves, D_full,
                           max_cluster_number=max_cluster_number,
                           max_cluster_content=max_cluster_content,
                           force_k=force_k,
                           flg_strict_split=flg_strict_split)

    # Convert indices -> labels for presentation
    nested = _map_indices_to_labels(idx_tree, labels)

    # Output selection
    if output_format.lower() == "newick":
        out: Union[List[object], List[str], str] = nested_list_to_newick(nested)  # str
    elif output_format.lower() == "pathways":
        out = nested_list_to_pathways(nested)  # List[str]
    else:
        out = nested  # nested list

    # Optional file write
    if output_file:
        out_text = str(out)
        if isinstance(out, list) and out and isinstance(out[0], str) and output_format.lower() == "pathways":
            #out_text = "\n".join([("   " * (len(line.split(">")) - 3)) + line for line in out])
            out = sorted(out, key=lambda line: [int(v) for v in line.split(">")[1:-1]])
            out_text = "\n".join(out)
        if output_file:
            tools.saveTextFile(out_text, output_file)

    return out

# ----------------- Example -----------------
if __name__ == "__main__":
    # Example: 5 items, lower-triangle input (no diagonal in rows)
    matrix = [
        [2.0],
        [3.0, 4.0],
        [2.0, 1.0, 5.0],
        [7.0, 6.0, 6.5, 6.8],
    ]
    labels = ["A", "B", "C", "D", "E"]

    clusters = spectral_analysis(matrix, labels,
                                 output_file="example.txt",
                                 output_format="pathways",
                                 max_cluster_number=5,
                                 max_cluster_content=5,
                                 flg_strict_split=True)  # toggle here
    print(clusters)
