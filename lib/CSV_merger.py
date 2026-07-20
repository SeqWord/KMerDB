#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Tuple
from word_db import Filter as fl
import tools

def filter_words(matrix, filter_settings, sort_para, flg_common_words = False):
    oFilter = fl(filter_settings = filter_settings, sort_para = sort_para)
    words = matrix.columns.tolist()[1:]
    for i in range(len(words)):
        word = words[i]
        w, wlength, x, y = word.split("|")
        col = matrix.iloc[:, i + 1]
        dpw = pw = abs(col.var())

        if flg_common_words:
            pw, dpw = [-pw, -dpw]

        flg_added = oFilter.add(int(wlength), int(x), int(y), pw)
        if int(wlength) == 7 and int(x) == 34 and int(y) == 192 and not flg_added:
            print("OK", flg_added)
            5/0
        if flg_added:
            oFilter.set_data(int(wlength), int(x), int(y), dpw)

    filtered_words = oFilter.get_words(flg_includeData = False, flg_tostring = True)
    print("\t" + f"Initial matrix length = {len(words)}; after filtering = {len(filtered_words)}")
    print()
    return filtered_words

def select_columns_by_mean(
        data_df: pd.DataFrame,
        diverse_bottom: float = -0.5, diverse_top: float = 0.5,
        common_bottom: float = -1.0, common_top: float = 1.0,
        min_selected: int = 0
    ) -> Tuple[List[str], List[str]]:
    """
    Select columns by mean value ranges.

    - Diverse: mean in [diverse_bottom, diverse_top]
    - Common:  mean in (-inf, common_bottom] ∪ [common_top, +inf)

    If min_selected > 0:
      - If len(diverse_cols) < min_selected: replace with the top `min_selected`
        columns by smallest |mean| (ascending). If fewer columns exist, use all.
      - If len(common_cols)  < min_selected: replace with the top `min_selected`
        columns by largest |mean| (descending). If fewer columns exist, use all.
    """
    if diverse_bottom > diverse_top:
        raise ValueError("diverse_bottom must be <= diverse_top")
    if common_bottom >= common_top:
        raise ValueError("common_bottom must be < common_top")

    # Ensure numeric (coerce non-numeric to NaN; NaNs ignored by mean)
    data_df = data_df.apply(pd.to_numeric, errors="coerce")

    # Column means (skip NaN)
    means = data_df.mean(axis=0, skipna=True)

    # Primary rule-based selections
    diverse_cols = means[(means >= diverse_bottom) & (means <= diverse_top)].index.tolist()
    common_cols  = means[(means <= common_bottom) | (means >= common_top)].index.tolist()

    if min_selected > 0:
        abs_means = means.abs()

        # Drop NaN means for fallback ranking; keep deterministic order by also sorting by column name as tiebreaker
        # (pandas keeps index order stable for ties, but we can sort by name explicitly if desired).
        abs_means_no_nan = abs_means.dropna()

        # Diverse fallback: smallest |mean| first
        if len(diverse_cols) < min_selected:
            diverse_ranked = abs_means_no_nan.sort_values(ascending=True)
            # Take up to min_selected; if fewer columns exist, take all
            diverse_cols = diverse_ranked.index.tolist()[:min_selected]

        # Common fallback: largest |mean| first
        if len(common_cols) < min_selected:
            common_ranked = abs_means_no_nan.sort_values(ascending=False)
            # Take up to min_selected; if fewer columns exist, take all
            common_cols = common_ranked.index.tolist()[:min_selected]

    return diverse_cols, common_cols

def execute(input_folder: str, output_dir: str, pattern: str = "*.csv",
            filter_settings: dict = {}, sort_para: int = 0,
            encoding: str = "utf-8", delimiter: str = ",",
            diverse_bottom: float = -0.5, diverse_top: float = 0.5,
            common_bottom: float = -1.0, common_top: float = 1.0,
            prefix_with_filename: bool = False):
    """
    Process CSV files from input_folder:
      - First two columns are TITLE columns (kept only from the FIRST CSV).
      - Remaining columns are numeric DATA columns with values in {-2, -1, 1, 2}.
    """
    csv_paths = sorted(glob.glob(os.path.join(input_folder, pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {input_folder} (pattern: {pattern})")

    diverse_matrix = None
    common_matrix = None
    title_ref = None  # (DataFrame with first two columns)

    for idx, path in enumerate(csv_paths):
        file_stem = os.path.splitext(os.path.basename(path))[0]
        print(f"📄 Processing: {path}")

        df = pd.read_csv(path, header=0, encoding=encoding, sep=delimiter)
        if df.shape[1] < 3:
            raise ValueError(f"{path} must have at least 3 columns (2 titles + ≥1 data).")

        # Split title vs data columns
        title_cols = df.iloc[:, :2].copy()
        # Make sure title columns are pandas string dtype using DataFrame.astype (safe)
        t0, t1 = title_cols.columns[0], title_cols.columns[1]
        title_cols = title_cols.astype({t0: "string", t1: "string"})

        data_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")  # numeric-only

        # Validate row order consistency across files using BOTH title columns
        if idx == 0:
            title_ref = title_cols.copy()
        else:
            if not title_ref.equals(title_cols):
                raise ValueError(f"Row titles (first two columns) differ or are out of order in: {path}")

        diverse_cols, common_cols = select_columns_by_mean(
            data_df, diverse_bottom, diverse_top, common_bottom, common_top
        )

        df_diverse = data_df[diverse_cols].copy()
        df_common  = data_df[common_cols].copy()

        if prefix_with_filename:
            df_diverse.columns = [f"{file_stem}::{c}" for c in df_diverse.columns]
            df_common.columns  = [f"{file_stem}::{c}" for c in df_common.columns]

        if idx == 0:
            diverse_matrix = pd.concat([title_cols, df_diverse], axis=1)
            common_matrix  = pd.concat([title_cols, df_common], axis=1)
        else:
            if df_diverse.shape[1] > 0:
                diverse_matrix = pd.concat([diverse_matrix, df_diverse], axis=1)
            if df_common.shape[1] > 0:
                common_matrix = pd.concat([common_matrix, df_common], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    # === Optional filtering of overlapping words (DATA columns only) ===
    filter_settings = None
    if filter_settings:
        print("📄 Filter diverse words...")
        filtered_word_list = filter_words(matrix=diverse_matrix.iloc[:, 2:],
                                          filter_settings=filter_settings,
                                          sort_para=sort_para)
        keep_cols = list(diverse_matrix.columns[:2]) + \
                    [c for c in diverse_matrix.columns[2:] if c in set(filtered_word_list)]
        diverse_matrix = diverse_matrix.loc[:, keep_cols]

        print("📄 Filter common words...")
        filtered_word_list = filter_words(matrix=common_matrix.iloc[:, 2:],
                                          filter_settings=filter_settings,
                                          sort_para=sort_para,
                                          flg_common_words=True)
        keep_cols = list(common_matrix.columns[:2]) + \
                    [c for c in common_matrix.columns[2:] if c in set(filtered_word_list)]
        common_matrix = common_matrix.loc[:, keep_cols]

    # === Write outputs ===
    if diverse_matrix is not None:
        outpath = os.path.join(output_dir, "diverse_features.csv")
        diverse_matrix.to_csv(outpath, index=False, header=True, encoding=encoding)
        print(f"✅ Wrote: {outpath}")
    else:
        print("⚠️ No 'diverse' features found across all files.")

    if common_matrix is not None:
        outpath = os.path.join(output_dir, "common_features.csv")
        common_matrix.to_csv(outpath, index=False, header=True, encoding=encoding)
        print(f"✅ Wrote: {outpath}")
    else:
        print("⚠️ No 'common' features found across all files.")

def summarize_node(
        dirpath: str,
        _retrieve_value: Callable[[List[str], List[str]], pd.DataFrame],
        diverse_bottom: float = -0.5, diverse_top: float = 0.5,
        common_bottom: float = -1.0, common_top: float = 1.0,
        min_selected: int = 10, top_selected: int = 50,
        flg_relabel: bool = True
    ):
    """
    Summarize immediate subfolders of `dirpath` into:
      - common_features.csv
      - diverse_features.csv

    If the union (intersection) of columns becomes too small (< min_matrix_size),
    extend it by selecting additional markers (by prevalence across subfolders),
    add them as new columns pre-filled with "0", and batch-fill per subfolder
    via `_retrieve_value(acc_list, marker_list)` which MUST return a DataFrame
    with rows in the same order as `acc_list` and columns in the same order as
    `marker_list`.
    """

    def csv_paths(base: str) -> Tuple[str, str]:
        return (os.path.join(base, "common_features.csv"),
                os.path.join(base, "diverse_features.csv"))

    # Minimal matrix size
    min_matrix_size = 3 * top_selected if top_selected else 100
    
    # -------------------------
    # Discover subdirectories
    # -------------------------
    subdirs = sorted(
        [os.path.join(dirpath, d) for d in os.listdir(dirpath)
         if os.path.isdir(os.path.join(dirpath, d))]
    )
    if not subdirs:
        return

    # Single subfolder → copy both CSVs up
    if len(subdirs) == 1:
        csrc, dsrc = csv_paths(subdirs[0])
        if os.path.isfile(csrc):
            tools.copy2(csrc, os.path.join(dirpath, "common_features.csv"))
        if os.path.isfile(dsrc):
            tools.copy2(dsrc, os.path.join(dirpath, "diverse_features.csv"))
        return

    # -------------------------
    # Load originals & build union by intersection
    # -------------------------
    matrix: pd.DataFrame | None = None
    contributors: List[str] = []
    matrix_titles: List[str] | None = None
    originals: Dict[str, pd.DataFrame] = {}  # subfolder_name -> original DF (possibly relabeled)

    for sub in subdirs:
        csrc, _ = csv_paths(sub)
        if not os.path.isfile(csrc):
            continue

        df = pd.read_csv(csrc, header=0)
        if df.shape[1] <= 2:
            continue

        # Optional relabel of second title column to subfolder name
        sub_name = os.path.basename(os.path.normpath(sub))
        if flg_relabel:
            col1 = df.columns[1]
            df = df.astype({col1: "string"})
            df[col1] = sub_name

        # Ensure title columns are strings
        t0, t1 = df.columns[0], df.columns[1]
        df = df.astype({t0: "string", t1: "string"})

        # Data columns to numeric
        data_cols = df.columns[2:].tolist()
        df[data_cols] = df[data_cols].apply(pd.to_numeric, errors="coerce")

        # Keep original for later batch retrieval
        originals[sub_name] = df.copy()

        # Build union by intersection of columns
        if matrix is None:
            matrix = df.copy()
            contributors.append(sub_name)
            matrix_titles = matrix.columns[:2].tolist()
        else:
            assert matrix_titles is not None
            current_order = matrix.columns[2:].tolist()
            sub_data_cols = df.columns[2:].tolist()
            intersect = [c for c in current_order if c in sub_data_cols]
            if not intersect:
                continue

            # Align title column names if needed
            if df.columns[0] != matrix_titles[0] or df.columns[1] != matrix_titles[1]:
                new_cols = matrix_titles + df.columns[2:].tolist()
                df = df.copy()
                df.columns = new_cols

            matrix = pd.concat(
                [matrix[matrix_titles + intersect], df[matrix_titles + intersect]],
                axis=0, ignore_index=True
            )
            contributors.append(sub_name)

    # If union couldn't be built, bubble up one child's files
    if matrix is None or len(contributors) < 2:
        fallback_sub = contributors[0] if contributors else os.path.basename(os.path.normpath(subdirs[0]))
        csrc, dsrc = csv_paths(os.path.join(dirpath, fallback_sub))
        if os.path.isfile(csrc):
            tools.copy2(csrc, os.path.join(dirpath, "common_features.csv"))
        if os.path.isfile(dsrc):
            tools.copy2(dsrc, os.path.join(dirpath, "diverse_features.csv"))
        return

    # -------------------------
    # Expand union if data columns < min_matrix_size
    # -------------------------
    titles = matrix.columns[:2].tolist()
    accession_col, subfolder_col = titles[0], titles[1]
    current_data_cols = matrix.columns[2:].tolist()
    n_current = len(current_data_cols)

    if n_current < min_matrix_size:
        # Build set of all markers from originals
        all_markers = set()
        for odf in originals.values():
            all_markers.update(odf.columns[2:].tolist())

        # Presence table: rows=markers, cols=contributors, values in {0,1}
        contrib_cols = contributors[:]  # keep union order
        presence_rows = sorted(all_markers)
        presence = pd.DataFrame(0, index=presence_rows, columns=contrib_cols, dtype=int)
        for sub_name, odf in originals.items():
            if sub_name not in presence.columns:
                continue
            cols = odf.columns[2:].tolist()
            presence.loc[[c for c in cols if c in presence.index], sub_name] = 1

        # Remove markers already in union and all-ones (universal) markers
        if n_current > 0:
            presence.drop(index=[c for c in current_data_cols if c in presence.index],
                          inplace=True, errors="ignore")
        presence["_sum"] = presence.sum(axis=1)
        n_contrib = len(contrib_cols)
        presence = presence[presence["_sum"] < n_contrib].sort_values("_sum", ascending=False)

        # How many more columns needed?
        need = max(0, min_matrix_size - n_current)
        
        if need > 0 and not presence.empty:
            chosen = presence.index.tolist()[:need]

            # 1) Add chosen markers as new columns, fill with "0"
            for marker in chosen:
                matrix[marker] = "0"

            # 2) Batch-retrieve by subfolder for the chosen markers only
            for sub_name in contrib_cols:
                block_mask = (matrix[subfolder_col].astype(str) == str(sub_name))
                if not block_mask.any():
                    continue

                acc_list = matrix.loc[block_mask, accession_col].astype(str).tolist()
                marker_list = [m for m in chosen if m in presence.index]
                if not acc_list or not marker_list:
                    continue

                # Retrieve a DataFrame with index=acc_list, columns=marker_list
                try:
                    block_df = _retrieve_value(acc_list, marker_list)
                except Exception:
                    # If retrieval fails, keep "0" as placeholder
                    continue

                if not isinstance(block_df, pd.DataFrame):
                    # Enforce expected type
                    continue

                # Reindex to the exact order we expect (no surprises)
                block_df = block_df.reindex(index=acc_list, columns=marker_list)

                # Assign values into the union matrix in one shot
                sub_idx = matrix.index[block_mask]
                if len(sub_idx) == len(acc_list):
                    # Fast path: same order
                    matrix.loc[sub_idx, marker_list] = block_df.values
                else:
                    # Align by accession name (slower but safe)
                    acc_to_row = {str(matrix.at[i, accession_col]): i for i in sub_idx}
                    for acc in block_df.index:
                        ridx = acc_to_row.get(str(acc))
                        if ridx is not None:
                            matrix.loc[ridx, marker_list] = block_df.loc[acc, marker_list].values

            # Clean helper column if present
            if "_sum" in presence.columns:
                presence.drop(columns=["_sum"], inplace=True, errors="ignore")

    # -------------------------
    # Downstream selection (diverse/common) on the (possibly expanded) union
    # -------------------------
    titles = matrix.columns[:2].tolist()
    matrix_data = matrix.iloc[:, 2:].copy().apply(pd.to_numeric, errors="coerce")

    # User-supplied selector (must exist in your environment)
    diverse_cols, common_cols = select_columns_by_mean(
        data_df=matrix_data,
        diverse_bottom=diverse_bottom, diverse_top=diverse_top,
        common_bottom=common_bottom, common_top=common_top,
        min_selected=min_selected
    )

    diverse_out_df = (pd.concat([matrix[titles], matrix[diverse_cols]], axis=1)
                      if diverse_cols else matrix[titles].copy())
    common_out_df  = (pd.concat([matrix[titles], matrix[common_cols]],  axis=1)
                      if common_cols  else matrix[titles].copy())

    # Save outputs
    diverse_out_df.to_csv(os.path.join(dirpath, "diverse_features.csv"), index=False)
    common_out_df.to_csv(os.path.join(dirpath, "common_features.csv"),  index=False)
