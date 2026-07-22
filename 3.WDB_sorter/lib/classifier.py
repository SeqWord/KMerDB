import os
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def main(inpath,
         file_name="diverse_features.csv",
         outpath="",
         p_value=None,
         significance=None,
         min_selected=10,
         top_selected=100,
         long_word_prefered: bool = False):
    """
    Rank columns by:
      1) FDR asc
      2) p-value asc
      3) int(title.split('|')[1]) asc; title is like 'CGTCA|5|14|52'
      4) title (asc by default; DESC if long_word_prefered=True)

    If filters yield < min_selected, top-up from the global ranking.
    If top_selected > 0, trim but never below min_selected.
    """
    file_path = os.path.join(inpath, file_name)
    if not os.path.exists(file_path):
        print(f"\tError with classifier, file {file_path} does not exists!")
        return []

    if outpath == "":
        outpath = inpath

    df = pd.read_csv(file_path)

    # Clean labels in 2nd column (group labels)
    df = df[df.iloc[:, 1].notna()]
    df = df[df.iloc[:, 1].astype(str).str.strip() != ""]
    df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip()

    labels = df.iloc[:, 1]
    data = df.iloc[:, 2:].astype(int)  # values in {-2, -1, 1, 2}

    # Chi-square per column
    results = []
    for i, col in enumerate(data.columns):
        contingency = pd.crosstab(labels, data[col])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
            except ValueError:
                p = float('nan')
        else:
            p = float('nan')
        results.append((i + 1, col, p))

    result_df = pd.DataFrame(results, columns=["Column", "Location", "p-value"])

    # FDR
    pvals = result_df["p-value"].values
    valid = ~pd.isnull(pvals)
    result_df["FDR p-value"] = float('nan')
    result_df["Significant (FDR<0.05)"] = False
    if valid.any():
        _, fdr_p, _, _ = multipletests(pvals[valid], alpha=0.05, method='fdr_bh')
        result_df.loc[valid, "FDR p-value"] = fdr_p
        result_df.loc[valid, "Significant (FDR<0.05)"] = fdr_p < 0.05

    # Tie-break key: int(title.split('|')[1]); plus title string for final tie-break
    def _tie_key(title: str) -> int:
        try:
            return int(str(title).split("|")[1])
        except Exception:
            return 10**9  # push malformed to end

    result_df["__tie__"] = result_df["Location"].map(_tie_key)

    # Global ranking (respect long_word_prefered for title direction)
    # Sort keys: FDR, p, __tie__, Location
    ascending_flags = [True, True, True, not long_word_prefered]
    ranked_df = result_df.sort_values(
        ["FDR p-value", "p-value", "__tie__", "Location"],
        na_position="last",
        ascending=ascending_flags
    ).reset_index(drop=True)

    # Apply filters
    filtered_df = ranked_df.copy()
    if p_value is not None:
        filtered_df = filtered_df[filtered_df["p-value"] <= p_value]
    if significance:
        filtered_df = filtered_df[filtered_df["Significant (FDR<0.05)"]]

    # Top-up to min_selected if needed
    if min_selected and min_selected > 0 and len(filtered_df) < min_selected:
        need = min_selected - len(filtered_df)
        remaining = ranked_df[~ranked_df["Location"].isin(filtered_df["Location"])]
        filtered_df = pd.concat([filtered_df, remaining.head(need)], ignore_index=True)

    # Reorder filtered_df to match global ranking order exactly
    order_map = {loc: i for i, loc in enumerate(ranked_df["Location"].tolist())}
    filtered_df = (
        filtered_df.assign(__ord__=filtered_df["Location"].map(order_map))
                   .sort_values("__ord__")
                   .drop(columns="__ord__")
    )

    # Save the filtered stats table (drop helper)
    os.makedirs(outpath, exist_ok=True)
    out_file = os.path.join(outpath, "chi2_fdr_results.csv")
    save_df = filtered_df.drop(columns=[c for c in ["__tie__"] if c in filtered_df.columns])
    save_df.to_csv(out_file, index=False)

    # Build the return list
    grouped = data.groupby(labels)
    selected = []
    for _, row in filtered_df.iterrows():
        col = row["Location"]
        means_per_group = grouped[col].mean().to_dict()
        selected.append({
            "title": col,
            "p": float(row["p-value"]) if pd.notnull(row["p-value"]) else float('nan'),
            "FDR": float(row["FDR p-value"]) if pd.notnull(row["FDR p-value"]) else float('nan'),
            "groups": means_per_group
        })

    # Apply top_selected (never below min_selected)
    if top_selected and top_selected > 0 and len(selected) > top_selected:
        cut = max(top_selected, min_selected or 0)
        selected = selected[:cut]

    print(f"Selected {len(selected)} columns "
          f"(min_selected={min_selected}, top_selected={top_selected}, "
          f"long_word_prefered={long_word_prefered}).")
    return selected
