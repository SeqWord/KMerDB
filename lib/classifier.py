import os
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------- I/O ----------

def _load_input(inpath: str, file_name: str) -> pd.DataFrame | None:
    file_path = os.path.join(inpath, file_name)
    if not os.path.exists(file_path):
        print(f"\tError with classifier, file {file_path} does not exists!")
        return None
    return pd.read_csv(file_path)

def _ensure_outdir(outpath: str) -> str:
    if not outpath:
        raise ValueError("outpath must be a non-empty path")
    os.makedirs(outpath, exist_ok=True)
    return outpath

def _save_filtered_table(df: pd.DataFrame, outpath: str, name: str = "chi2_fdr_results.csv") -> str:
    out_file = os.path.join(outpath, name)
    df.to_csv(out_file, index=False)
    return out_file

# ---------- Cleaning / preparation ----------

def _clean_labels_and_values(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return (labels, numeric_data_frame)"""
    label_col = df.columns[1]

    # Clean labels (string dtype, drop NaNs and empties)
    df = df[df[label_col].notna()].astype({label_col: "string"})
    df[label_col] = df[label_col].str.strip()
    df = df[df[label_col] != ""]
    labels = df[label_col]

    # Convert data block (cols 3..end) to integers in {-2,-1,1,2}, coerce invalid to 0
    data = (
        df.iloc[:, 2:]
          .apply(pd.to_numeric, errors="coerce")
          .replace([float("inf"), float("-inf")], 0)
          .fillna(0)
          .astype(int)
    )
    return labels, data

# ---------- Statistics ----------

def _chi2_by_column(labels: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns ['Column', 'Location', 'p-value']"""
    rows = []
    for i, col in enumerate(data.columns):
        contingency = pd.crosstab(labels, data[col])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        rows.append((i + 1, col, p))
    return pd.DataFrame(rows, columns=["Column", "Location", "p-value"])

def _fdr_correction(result_df: pd.DataFrame) -> pd.DataFrame:
    """Add 'FDR p-value' and 'Significant (FDR<0.05)' columns."""
    pvals = result_df["p-value"].values
    valid = ~pd.isnull(pvals)
    result_df = result_df.copy()
    result_df["FDR p-value"] = float("nan")
    result_df["Significant (FDR<0.05)"] = False
    if valid.any():
        _, fdr_p, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        result_df.loc[valid, "FDR p-value"] = fdr_p
        result_df.loc[valid, "Significant (FDR<0.05)"] = fdr_p < 0.05
    return result_df

# ---------- Ranking & filtering ----------

def _tie_key(title: str) -> int:
    try:
        return int(str(title).split("|")[1])
    except Exception:
        return 10**9  # push malformed to the end

def _rank_global(result_df: pd.DataFrame, long_word_prefered: bool) -> pd.DataFrame:
    """Sort by FDR asc, p asc, int(part[1]) asc, title asc/desc (desc if long_word_prefered)."""
    ranked = result_df.copy()
    ranked["__tie__"] = ranked["Location"].map(_tie_key)
    ascending_flags = [True, True, True, not long_word_prefered]
    ranked = ranked.sort_values(
        ["FDR p-value", "p-value", "__tie__", "Location"],
        na_position="last",
        ascending=ascending_flags
    ).reset_index(drop=True)
    return ranked

def _apply_filters(ranked_df: pd.DataFrame,
                   p_value: float | None,
                   significance: bool | None,
                   min_selected: int,
                   top_selected: int | None) -> pd.DataFrame:
    """Filter, top-up to min_selected, and keep global order."""
    filtered = ranked_df.copy()
    if p_value is not None:
        filtered = filtered[filtered["p-value"] <= p_value]
    if significance:
        filtered = filtered[filtered["Significant (FDR<0.05)"]]

    # Top-up to min_selected
    if min_selected and min_selected > 0 and len(filtered) < min_selected:
        need = min_selected - len(filtered)
        remaining = ranked_df[~ranked_df["Location"].isin(filtered["Location"])]
        filtered = pd.concat([filtered, remaining.head(need)], ignore_index=True)

    # Preserve exact global ordering
    order_map = {loc: i for i, loc in enumerate(ranked_df["Location"].tolist())}
    filtered = (
        filtered.assign(__ord__=filtered["Location"].map(order_map))
                .sort_values("__ord__")
                .drop(columns="__ord__")
    )

    # Apply top_selected but never below min_selected
    if top_selected and top_selected > 0 and len(filtered) > top_selected:
        cut = max(top_selected, min_selected or 0)
        filtered = filtered.iloc[:cut, :]

    return filtered

# ---------- Model building ----------

def _groups_as_list(grouped: pd.core.groupby.generic.DataFrameGroupBy, col: str) -> list[list[object]]:
    """
    Return [[group_label, mean_value], ...] sorted by group_label as string.
    (Replaces dict return.)
    """
    means = grouped[col].mean()
    # Convert to list of [label, mean] (labels kept as strings)
    items = [[str(idx), float(val)] for idx, val in means.items()]
    # Stable order by label string (optional)
    items.sort(key=lambda x: x[0])
    return items

def _build_decision_model(filtered_df: pd.DataFrame,
                          labels: pd.Series,
                          data: pd.DataFrame) -> list[dict]:
    grouped = data.groupby(labels)
    selected = []
    for _, row in filtered_df.iterrows():
        col = row["Location"]
        group_list = _groups_as_list(grouped, col)  # <-- list of [label, mean]
        selected.append({
            "title": col,
            "p": float(row["p-value"]) if pd.notnull(row["p-value"]) else float("nan"),
            "FDR": float(row["FDR p-value"]) if pd.notnull(row["FDR p-value"]) else float("nan"),
            "groups": group_list
        })
    return selected

# ---------- Public entry ----------

def main(inpath: str,
         file_name: str = "diverse_features.csv",
         outpath: str = "",
         p_value: float | None = None,
         significance: bool | None = None,
         min_selected: int = 10,
         top_selected: int = 100,
         long_word_prefered: bool = False) -> tuple[list[dict], pd.DataFrame, int]:
    """
    Rank columns by:
      1) FDR asc
      2) p-value asc
      3) int(title.split('|')[1]) asc; title is like 'CGTCA|5|14|52'
      4) title (asc by default; DESC if long_word_prefered=True)

    If filters yield < min_selected, top-up from the global ranking.
    If top_selected > 0, trim but never below min_selected.

    Returns:
      decision_model: list of dicts, each like:
        {
          'title': <column title>,
          'p': <raw p-value or NaN>,
          'FDR': <BH corrected p or NaN>,
          'groups': [[<group_label>, <mean over rows in that group>], ...]
        },
      df pd.DataFrame matrix,
      median_cluster_size: int
        Median of label frequencies in the `labels` array, i.e., median size of taxa clusters.
    """
    df = _load_input(inpath, file_name)
    if df is None:
        return ([], 0)

    if not outpath:
        outpath = inpath
    _ensure_outdir(outpath)

    # Prep data
    labels, data = _clean_labels_and_values(df)

    # ---- NEW: compute median cluster size over label counts ----
    # labels can be a numpy array, list, or pandas Series; Counter handles all.
    label_counts = Counter(list(labels))
    median_cluster_size = int(np.median(list(label_counts.values()))) if label_counts else 0

    # Stats
    result_df = _chi2_by_column(labels, data)
    result_df = _fdr_correction(result_df)

    # Ranking & filtering
    ranked_df = _rank_global(result_df, long_word_prefered=long_word_prefered)
    filtered_df = _apply_filters(
        ranked_df, p_value=p_value, significance=significance,
        min_selected=min_selected, top_selected=top_selected
    )

    # Save filtered stats (drop helper col if present)
    to_save = filtered_df.drop(columns=[c for c in ["__tie__"] if c in filtered_df.columns], errors="ignore")
    _save_filtered_table(to_save, outpath)

    # Build model with groups as list-of-pairs
    decision_model = _build_decision_model(filtered_df, labels, data)

    print(
        f"Selected {len(decision_model)} columns "
        f"(min_selected={min_selected}, top_selected={top_selected}, "
        f"long_word_prefered={long_word_prefered}); "
        f"median cluster size = {median_cluster_size}."
    )
    return decision_model, df, median_cluster_size
