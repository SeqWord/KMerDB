from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
"""
How this fits your architecture

Traversal uses your child_node dict; no reliance on your address strings or get_distance(...).

Branch lengths are read from each node’s branch_length and accumulated to get true depths from the root.

Shortcuts: every committed cluster is exposed at the root as Nick(title, object=node_root) under self.short_cuts[label]. 
That gives you fast navigation to the coalesced zones without altering the original links.

Fan-out control: any cluster whose tip count exceeds target_fanout is split internally (greedy by child subtrees) until 
each subcluster is ≤ target_fanout tips. You can swap oversize_split for a more sophisticated splitter later (e.g., 
depth median or patristic radius).

Adaptive sectors: quantile binning yields narrow sectors where nodes are dense and wide sectors where they’re sparse — 
exactly the behavior you asked for. If you want even more sensitivity, you can replace quantiles with Bayesian Blocks 
later; the rest of the code won’t change much.

Notes / extensions

If you prefer the reduced tree to be materialized (physically collapsed), we can add an in_place_collapse=True mode 
that replaces each subtree by a single Intermediate_Node carrying metadata and rewires child_node accordingly. 
I kept the non-destructive shortcut approach to be safe.

You can seed the number of sectors by passing sectors=... (e.g., equals the number of levels you want), otherwise 
it’s chosen from the number of tips and target_fanout.

"""
@dataclass
class Cluster:
    """Lightweight record of a collapsed in-sector subtree."""
    label: str
    root_title: str
    sector: Tuple[float, float]          # (a, b) depth sector
    member_tips: List[str]               # leaf titles under this cluster
    member_internal: List[str]           # internal node titles under this cluster
    min_depth: float
    max_depth: float
    size_tips: int
    size_nodes: int
    
class Nick:
    self.__init__(self, title str, obj):
        self.title = title
        self.objext = obj

# --- Add this method inside your Root class ---
def coalesce(self,
             target_fanout: int = 5,
             sectors: Optional[int] = None,
             max_sectors: int = 64,
             method: str = "quantile",
             in_place_shortcuts: bool = True,
             shortcut_prefix: str = "CLUST_",
             oversize_split: str = "greedy") -> List[Cluster]:
    """
    Reduce a dichotomic dendrogram by clustering subtrees inside
    depth 'sectors' (variable-width bins along distance-from-root).

    Parameters
    ----------
    target_fanout : int
        Desired ~number of descendant tips per cluster (≈5 by default).
    sectors : Optional[int]
        If None, chosen adaptively from tree size. Otherwise, fixed number
        of depth sectors (variable-width by equal-count quantiles).
    max_sectors : int
        Upper bound when auto-selecting sectors.
    method : str
        Currently only 'quantile' (equal-count bins) is implemented.
    in_place_shortcuts : bool
        If True, create Nick-based shortcuts at the root for each cluster.
        The original topology is not destroyed.
    shortcut_prefix : str
        Prefix for new shortcut titles (e.g., 'CLUST_001').
    oversize_split : str
        Strategy to split clusters with too many tips: 'greedy' (by children).

    Returns
    -------
    List[Cluster]
        Structured info about each cluster that was created.
    """
    # --------- 1) Traverse to collect topology, depths, parents ----------
    # We'll do one DFS to compute depth-from-root and parent map
    parent: Dict[Any, Optional[Any]] = {self: None}
    depth: Dict[Any, float] = {self: 0.0}
    preorder: List[Any] = []

    def dfs(node, d):
        preorder.append(node)
        depth[node] = d
        if hasattr(node, "child_node") and node.child_node:
            for child in node.child_node.values():
                parent[child] = node
                # Accumulate distance using each child's own branch_length
                child_bl = getattr(child, "branch_length", 0.0) or 0.0
                dfs(child, d + child_bl)

    dfs(self, 0.0)

    # Gather all nodes and leaves (tips)
    def is_leaf(n):
        return not (hasattr(n, "child_node") and n.child_node)

    all_nodes = preorder
    leaves = [n for n in all_nodes if is_leaf(n)]
    internals = [n for n in all_nodes if not is_leaf(n)]

    # --------- 2) Postorder to compute min/max descendant depths ----------
    min_d: Dict[Any, float] = {}
    max_d: Dict[Any, float] = {}

    # Build children list helper
    children: Dict[Any, List[Any]] = {}
    for n in all_nodes:
        children[n] = list(n.child_node.values()) if hasattr(n, "child_node") and n.child_node else []

    # Postorder traversal (reverse of preorder with a check)
    for n in reversed(all_nodes):
        if not children[n]:  # leaf
            min_d[n] = max_d[n] = depth[n]
        else:
            md = min([min_d[c] for c in children[n]] + [depth[n]])
            xd = max([max_d[c] for c in children[n]] + [depth[n]])
            min_d[n], max_d[n] = md, xd

    # --------- 3) Build the depth axis and choose sectors ----------
    depth_samples = [depth[n] for n in all_nodes]  # nodes density along axis
    N = len(depth_samples)

    if sectors is None:
        # Heuristic: aim ~ (total_tips / target_fanout) clusters; cap by max_sectors
        approx_clusters = max(1, round(len(leaves) / max(1, target_fanout)))
        sectors = min(max_sectors, max(1, approx_clusters))
    sectors = max(1, sectors)

    # Quantile edges -> variable-width bins w/ roughly equal counts of nodes
    # Add tiny jitter to collapse duplicate edges if depths are identical
    qs = [i / sectors for i in range(sectors + 1)]
    sorted_depths = sorted(depth_samples)
    def qtile(q):
        idx = int(round(q * (N - 1)))
        return sorted_depths[idx]

    edges = [qtile(q) for q in qs]
    # Ensure strictly increasing edges (handle flat regions)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12  # tiny epsilon to separate

    sectors_ab = list(zip(edges[:-1], edges[1:]))

    # --------- 4) Find maximal in-sector subtrees (monophyletic) ----------
    def inside_sector(n, a, b):
        # Entire subtree is within sector if all descendants lie in [a,b]
        return (min_d[n] >= a) and (max_d[n] <= b)

    clusters: List[Cluster] = []
    cluster_ix = 1

    # Helper: collect tips/internal under a node
    def collect_members(n) -> Tuple[List[Any], List[Any]]:
        tips, inns = [], []
        stack = [n]
        while stack:
            x = stack.pop()
            if is_leaf(x):
                tips.append(x)
            else:
                inns.append(x)
                stack.extend(children[x])
        return tips, inns

    # Split an oversized cluster by greedily grouping children until tip cap
    def split_oversize(n, cap) -> List[Any]:
        """Return a list of subtree roots each with <= cap tips."""
        # If already small enough, keep as one
        tips, _ = collect_members(n)
        if len(tips) <= cap:
            return [n]
        # Greedy: start from this node's children and group them
        groups = []
        cur_group = []
        cur_count = 0
        for ch in children[n]:
            ch_tips, _ = collect_members(ch)
            tcount = len(ch_tips)
            # If single child too big, recurse on it
            if tcount > cap and children[ch]:
                groups.extend(split_oversize(ch, cap))
                continue
            if cur_count + tcount <= cap:
                cur_group.append(ch)
                cur_count += tcount
            else:
                # finalize current group -> make a virtual root by choosing the MRCA (n),
                # but we'll just return the grouped children as separate cluster roots
                if cur_group:
                    # return each child as its own cluster root
                    groups.extend(cur_group)
                # start new group with current child
                cur_group = [ch]
                cur_count = tcount
        if cur_group:
            groups.extend(cur_group)
        return groups

    # Create a Cluster record and optional Nick shortcut
    def commit_cluster(node_root, a, b):
        nonlocal cluster_ix
        tips, inns = collect_members(node_root)
        # Enforce size cap
        if len(tips) > target_fanout:
            parts = split_oversize(node_root, target_fanout)
            for p in parts:
                commit_cluster(p, a, b)
            return
        label = f"{shortcut_prefix}{cluster_ix:03d}"
        cl = Cluster(
            label=label,
            root_title=getattr(node_root, "title", str(node_root)),
            sector=(a, b),
            member_tips=[getattr(t, "title", str(t)) for t in tips],
            member_internal=[getattr(x, "title", str(x)) for x in inns if x is not node_root],
            min_depth=min_d[node_root],
            max_depth=max_d[node_root],
            size_tips=len(tips),
            size_nodes=len(tips) + len(inns),
        )
        clusters.append(cl)
        if in_place_shortcuts:
            # Expose this cluster via a Nick at the root
            # Assumes you have Nick(title=..., object=...)
            self.short_cuts[label] = Nick(title=label, obj=node_root)
        cluster_ix += 1

    # For each sector, find maximal roots inside it
    for (a, b) in sectors_ab:
        # Candidates that are fully inside
        candidates = [n for n in all_nodes if inside_sector(n, a, b)]
        # Keep only those whose parent is not fully inside (maximality)
        for n in candidates:
            p = parent.get(n, None)
            if p is None or not inside_sector(p, a, b):
                commit_cluster(n, a, b)

    return clusters

if __name__ == "__main__":
    # inside your code after the tree is built
    clusters = root.coalesce(target_fanout=5, sectors=None, in_place_shortcuts=True)
    for c in clusters:
        print(c.label, c.size_tips, c.sector, "->", c.member_tips[:5], "...")
    # Navigate a cluster quickly via shortcut:
    some = root.short_cuts["CLUST_003"].object   # the subtree root of that cluster
    
