#!/usr/bin/env python3
import os
import glob
import pandas as pd
from typing import Tuple, List
from word_db import Filter as fl

def filter_words(matrix, filter_settings, sort_para, flg_common_words = False):
    oFilter = fl(filter_settings = filter_settings, sort_para = sort_para)
    words = matrix.columns.tolist()[1:]
    for i in range(len(words)):
        word = words[i]
        w, wlength, x, y = word.split("|")
        # Select values from the respective column
        col = matrix.iloc[:, i + 1]
        #dpw = pw = abs(col.mean())
        #dpw = pw = abs(col.abs().mean())
        dpw = pw = abs(col.var())
        
        if flg_common_words:
            pw, dpw = [-pw, -dpw]
            
        flg_added = oFilter.add(int(wlength), int(x), int(y), pw)
        if int(wlength) == 7 and int(x) ==34 and int(y) == 192 and not flg_added:
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
    ) -> Tuple[List[str], List[str]]:
    """
    Select columns by mean value ranges.

    - Diverse: mean in [diverse_bottom, diverse_top]
    - Common:  mean in (-inf, common_bottom] ∪ [common_top, +inf)
    DEFAULT:
      - diverse: mean in [-0.5, 0.5]
      - common:  mean in [-2, -1.5] U [1.5, 2]
    

    Expects `data_df` to contain only numeric data columns (no title cols).
    """
    if diverse_bottom > diverse_top:
        raise ValueError("diverse_bottom must be <= diverse_top")
    if common_bottom >= common_top:
        raise ValueError("common_bottom must be < common_top")

    # Ensure numeric (coerce non-numeric to NaN; NaNs ignored by mean)
    data_df = data_df.apply(pd.to_numeric, errors="coerce")

    means = data_df.mean(axis=0, skipna=True)

    diverse_cols = means[(means >= diverse_bottom) & (means <= diverse_top)].index.tolist()
    common_cols  = means[(means <= common_bottom) | (means >= common_top)].index.tolist()

    return diverse_cols, common_cols

def execute(input_folder: str, output_dir: str = ".", pattern: str = "*.csv",
            filter_settings: dict = {}, sort_para: int = 0,
            encoding: str = "utf-8", delimiter: str = ",",
            diverse_bottom: float = -0.5, diverse_top: float = 0.5,
            common_bottom: float = -1.0, common_top: float = 1.0,
            prefix_with_filename: bool = False):
    """
    Process CSV files from input_folder:
      - First two columns are TITLE columns (kept only from the FIRST CSV).
      - Remaining columns are numeric DATA columns with values in {-2, -1, 1, 2}.
      - For each file, partition DATA columns into 'diverse' (mean in [-0.5, 0.5])
        and 'common' (mean in [-2,-1.5] U [1.5,2]).
      - Initialize diverse/common matrices from the FIRST CSV as:
            [title_col1, title_col2, selected_data_cols]
        For subsequent files, concatenate ONLY the selected DATA columns to the right.
      - Save results to diverse_features.csv and common_features.csv

    Filtering step (optional):
      - If filter_settings provided, apply filter_words() to DATA columns only,
        keeping the two title columns intact.
    """
    csv_paths = sorted(glob.glob(os.path.join(input_folder, pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {input_folder} (pattern: {pattern})")

    diverse_matrix = None
    common_matrix = None
    # We’ll use the TWO TITLE COLUMNS from the first CSV as the row/ordering reference
    title_ref = None  # (DataFrame with first two columns)

    for idx, path in enumerate(csv_paths):
        file_stem = os.path.splitext(os.path.basename(path))[0]
        print(f"📄 Processing: {path}")

        # Read whole CSV (no index_col) so the first two columns remain as columns
        df = pd.read_csv(path, header=0, encoding=encoding, sep=delimiter)

        if df.shape[1] < 3:
            raise ValueError(f"{path} must have at least 3 columns (2 titles + ≥1 data).")

        # Split title vs data columns
        title_cols = df.iloc[:, :2].copy()
        data_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")  # numeric-only

        # Validate row order consistency across files using BOTH title columns
        if idx == 0:
            title_ref = title_cols.copy()
        else:
            if not title_ref.equals(title_cols):
                raise ValueError(f"Row titles (first two columns) differ or are out of order in: {path}")

        # Select columns by mean ranges
        diverse_cols, common_cols = select_columns_by_mean(data_df, 
            diverse_bottom, diverse_top,
            common_bottom, common_top,
        )

        # Subset dataframes to selected columns
        df_diverse = data_df[diverse_cols].copy()
        df_common  = data_df[common_cols].copy()

        # Optionally prefix data column names to avoid collisions
        if prefix_with_filename:
            df_diverse.columns = [f"{file_stem}::{c}" for c in df_diverse.columns]
            df_common.columns  = [f"{file_stem}::{c}" for c in df_common.columns]

        if idx == 0:
            # Initialize matrices with the two title cols + selected data
            diverse_matrix = pd.concat([title_cols, df_diverse], axis=1)
            common_matrix  = pd.concat([title_cols, df_common], axis=1)
        else:
            # Concatenate ONLY data columns to the right (keep first two title columns intact)
            if df_diverse.shape[1] > 0:
                diverse_matrix = pd.concat([diverse_matrix, df_diverse], axis=1)
            if df_common.shape[1] > 0:
                common_matrix = pd.concat([common_matrix, df_common], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    # === Optional filtering of overlapping words (DATA columns only) ===
    # TEMPORARILY SKIPPED unless filter_settings provided
    filter_settings = None
    if filter_settings:
        # Filter diverse
        print("📄 Filter diverse words...")
        # Expect filter_words(matrix=..., ...) to return a list of column names to keep
        filtered_word_list = filter_words(matrix=diverse_matrix.iloc[:, 2:],  # pass only data part
                                          filter_settings=filter_settings,
                                          sort_para=sort_para)
        keep_cols = list(diverse_matrix.columns[:2]) + \
                    [c for c in diverse_matrix.columns[2:] if c in set(filtered_word_list)]
        diverse_matrix = diverse_matrix.loc[:, keep_cols]

        # Filter common
        print("📄 Filter common words...")
        filtered_word_list = filter_words(matrix=common_matrix.iloc[:, 2:],  # pass only data part
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
        
def summarize_node(dirpath: str, 
        diverse_bottom: float = -0.5, diverse_top: float = 0.5,
        common_bottom: float = -1.0, common_top: float = 1.0, level_increment: float = 0.0,        
        flg_relabel: bool = True):
    """
    Visit all immediate subfolders of `dirpath` and summarize into:
      - common_features.csv
      - diverse_features.csv

    Files format in subfolders:
      • first 2 columns = title columns
      • remaining columns = numeric data columns

    Behavior:
      1) If exactly one subfolder → copy both CSVs up.
      2) Else:
         - Vertically concatenate subfolders' common_features.csv after
           intersecting data columns (order taken from the first contributor).
         - If flg_relabel is True, set the entire 2nd title column to subfolder name
           BEFORE concatenation (including the first contributor).
         - If <2 contributors → copy both CSVs from the only contributor; if none
           contributed (all empty), copy from the first subfolder.
         - Otherwise run select_columns_by_mean on concatenated data part and write
           both outputs with the two title columns preserved.
    """
    # --- discover immediate subfolders ---
    subdirs = sorted(
        [os.path.join(dirpath, d) for d in os.listdir(dirpath)
         if os.path.isdir(os.path.join(dirpath, d))]
    )
    if not subdirs:
        return  # nothing to do

    def csv_paths(base: str):
        return (os.path.join(base, "common_features.csv"),
                os.path.join(base, "diverse_features.csv"))

    # Single subfolder → copy both CSVs up
    if len(subdirs) == 1:
        csrc, dsrc = csv_paths(subdirs[0])
        if os.path.isfile(csrc):
            tools.copy2(csrc, os.path.join(dirpath, "common_features.csv"))
        if os.path.isfile(dsrc):
            tools.copy2(dsrc, os.path.join(dirpath, "diverse_features.csv"))
        return

    # --- build concatenation from common_features.csv across subfolders ---
    matrix = None                      # running concatenated DataFrame (titles + data)
    contributors: List[str] = []       # subdirs that actually contributed rows
    matrix_titles: List[str] = None    # first two column names from the first contributor

    for sub in subdirs:
        csrc, _ = csv_paths(sub)
        if not os.path.isfile(csrc):
            continue

        df = pd.read_csv(csrc, header=0)

        # Skip if "empty": only two title columns
        if df.shape[1] <= 2:
            continue

        # Optional relabel: set 2nd title column values to subfolder name
        if flg_relabel:
            sub_name = os.path.basename(os.path.normpath(sub))
            df.iloc[:, 1] = sub_name

        # Titles and data
        title_cols = df.columns[:2].tolist()
        data_cols  = df.columns[2:].tolist()

        # Coerce data columns to numeric
        df[data_cols] = df[data_cols].apply(pd.to_numeric, errors="coerce")

        if matrix is None:
            matrix = df.copy()
            contributors.append(sub)
            matrix_titles = matrix.columns[:2].tolist()
        else:
            # Intersect data columns; keep order from current matrix
            current_order = matrix.columns[2:].tolist()
            sub_data_cols = df.columns[2:].tolist()
            intersect = [c for c in current_order if c in sub_data_cols]
            if not intersect:
                continue

            # Ensure title column names match the running matrix to avoid misaligned concat
            if df.columns[0] != matrix_titles[0] or df.columns[1] != matrix_titles[1]:
                new_cols = matrix_titles + df.columns[2:].tolist()
                df = df.copy()
                df.columns = new_cols

            # Trim both to [title_cols + intersect] keeping order from matrix
            matrix = pd.concat(
                [matrix[matrix_titles + intersect], df[matrix_titles + intersect]],
                axis=0, ignore_index=True
            )
            contributors.append(sub)

    # Fallback: copy from a contributor (that actually added rows), else from first subfolder
    if matrix is None or len(contributors) < 2:
        source_sub = contributors[0] if contributors else subdirs[0]
        csrc, dsrc = csv_paths(source_sub)
        if os.path.isfile(csrc):
            tools.copy2(csrc, os.path.join(dirpath, "common_features.csv"))
        if os.path.isfile(dsrc):
            tools.copy2(dsrc, os.path.join(dirpath, "diverse_features.csv"))
        return

    # Compute selection from the concatenated matrix (data part only)
    matrix_data = matrix.iloc[:, 2:].copy()
    matrix_data = matrix_data.apply(pd.to_numeric, errors="coerce")

    # Calculate diverse and common K-mers for intermediate nodes    
    diverse_cols, common_cols = select_columns_by_mean(data_df=matrix_data,
        diverse_bottom=diverse_bottom, diverse_top=diverse_top,
        common_bottom=common_bottom, common_top=common_top
        )
    # Add common_inceement after each cycle
    diverse_bottom -= level_increment
    diverse_top += level_increment
    common_bottom -= level_increment
    common_top += level_increment

    # Save selected diverse/common with the two title columns
    titles = matrix.columns[:2].tolist()

    diverse_out_df = pd.concat([matrix[titles], matrix[diverse_cols]], axis=1) if diverse_cols else matrix[titles].copy()
    common_out_df  = pd.concat([matrix[titles], matrix[common_cols]],  axis=1) if common_cols  else matrix[titles].copy()

    diverse_out_df.to_csv(os.path.join(dirpath, "diverse_features.csv"), index=False)
    common_out_df.to_csv(os.path.join(dirpath, "common_features.csv"),  index=False)
