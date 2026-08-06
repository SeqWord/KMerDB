#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import os
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd

# =============================================================================
# Shared helpers (dataset-level, not bound to any particular model class)
# =============================================================================

def _order_preserving_label_map(y_str: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Preserve first-seen order of y string labels.
    Returns (titles, y_idx) where titles is list[str] and y_idx is int array.
    """
    seen: Dict[str, int] = {}
    titles: List[str] = []
    y_str = np.asarray(y_str).astype(str)
    for lbl in y_str:
        if lbl not in seen:
            seen[lbl] = len(titles)
            titles.append(lbl)
    y_idx = np.array([seen[lbl] for lbl in y_str], dtype=int)
    return titles, y_idx


def _drop_singletons(X: np.ndarray, y_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove samples that belong to classes with < 2 occurrences.
    Returns (X_filtered, y_filtered). If nothing to drop, returns inputs.
    """
    uniq, cnts = np.unique(y_idx, return_counts=True)
    keep_classes = set(uniq[cnts >= 2])
    if len(keep_classes) == len(uniq):
        return X, y_idx
    mask = np.array([cls in keep_classes for cls in y_idx], dtype=bool)
    return X[mask], y_idx[mask]


def _safe_split(
    X: np.ndarray,
    y_idx: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Avoid dropping singleton classes during split.
    If any class has <2 samples or unique classes <2 -> use all data for train/test.
    Else try stratified split; fallback to non-stratified.
    """
    from sklearn.model_selection import train_test_split
    uniq, cnts = np.unique(y_idx, return_counts=True)
    if uniq.size < 2 or np.any(cnts < 2):
        # Train on all and evaluate on the same set to preserve classes.
        return X, X, y_idx, y_idx
    try:
        return train_test_split(
            X, y_idx, test_size=test_size, stratify=y_idx, random_state=random_state
        )
    except ValueError:
        return train_test_split(
            X, y_idx, test_size=test_size, random_state=random_state
        )


def _set_classes_from_estimator(self_obj: Any, estimator: Any, titles: List[str]) -> None:
    """
    After fit, set self.classes_ to the actually trained classes in string form,
    using estimator.classes_ (indices) mapped through titles (strings).
    """
    if estimator is not None and hasattr(estimator, "classes_"):
        present_idx = np.asarray(estimator.classes_, dtype=int)
        self_obj.classes_ = np.array([titles[i] for i in present_idx], dtype=object)
    else:
        self_obj.classes_ = np.array(titles, dtype=object)


def _proba_to_predictions(proba: np.ndarray, titles_like: Union[List[str], np.ndarray]) -> Dict[str, float]:
    """
    Map a probability vector to {class_name: prob} safely.
    """
    titles = np.asarray(titles_like, dtype=object)
    n = min(len(proba), len(titles))
    return {str(titles[i]): float(proba[i]) for i in range(n)}


def _normalize_markers_common(feature_titles: List[str], markers: Any) -> dict:
    """
    Example numeric normalizer (not used by categorical models, but kept for completeness).
    Normalize user-supplied markers into a flat {feature: float_value} dict.
    """
    import numpy as _np
    import pandas as _pd

    def _to_float(x):
        if x is None:
            return 0.0
        try:
            v = float(x)
        except Exception:
            return 0.0
        if _np.isnan(v) or _np.isinf(v):
            return 0.0
        return float(_np.clip(v, -2.0, 2.0))

    norm: dict = {}

    # dict-like
    if hasattr(markers, "items"):
        for k, v in markers.items():
            norm[str(k)] = _to_float(v)
        return norm

    # pandas Series/DataFrame
    try:
        if isinstance(markers, _pd.Series):
            for k, v in markers.items():
                norm[str(k)] = _to_float(v)
            return norm
        if isinstance(markers, _pd.DataFrame):
            if markers.shape[0] == 0:
                return {}
            row = markers.iloc[0]
            for k, v in row.items():
                norm[str(k)] = _to_float(v)
            return norm
    except Exception:
        pass

    # iterable of pairs or list of names
    try:
        for item in markers:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                k, v = item
                norm[str(k)] = _to_float(v)
            else:
                norm[str(item)] = 1.0
        return norm
    except TypeError:
        return {str(markers): 1.0}


# =============================================================================
# Base class: MLearn (common functionality for categorical models)
# =============================================================================

class MLearn:
    """
    Base class for ML models on categorical marker matrices.

    Provides:
      - common storage (titles, feature_titles, pipeline, classes_, etc.)
      - feature-name normalization and mapping
      - symbol normalization: "", "nd", "nan" → ""
      - generic identify() implementation for models using OneHotEncoder pipelines
      - generic predict_proba()
    """

    def __init__(
        self,
        model: str,
        marker_style: str = "plain",    # Marker title style: plain (default) like 'marker_1' or combined (one-hot) like 'marker_1=0'
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False
    ) -> None:
        self.model = model
        self.processor = None   # Placeholder for a class creating matrix from row data
        self.qualifiers = {}    # Researved member for other application-specific variables
        self.marker_style = marker_style
        self.categorical = bool(categorical)
        self.min_coverage_warn = float(min_coverage_warn)
        self.raise_on_low_coverage = bool(raise_on_low_coverage)

        # Filled by train()
        self.titles: List[str] = []                  # class titles (original labels, ordered)
        self.pipeline: Any = None                    # sklearn pipeline
        self.classes_: np.ndarray | None = None      # estimator.classes_ mapped to titles
        self.training_metrics: dict | None = None    # accuracy, report, confusion matrix
        self.feature_titles: List[str] | None = None # original marker names (pre-encoding)
        self.feature_values: dict[str, list[str]]    # Dictionary of feature values per feature: self.feature_values["yp1108ms45"] -> ["128", "145", "nd", ""]
        self._norm_feature_map: Dict[str, str] | None = None  # normalized_name -> canonical training name

    # -------------------------- name & symbol helpers -------------------------

    @staticmethod
    def _norm_name(x: Any) -> str:
        """
        Normalize a feature name or marker key:
          - cast to string
          - strip spaces
          - lowercase
          - keep only letters and digits
        """
        s = str(x).strip().lower()
        return "".join(ch for ch in s if ch.isalnum())

    @staticmethod
    def _normalize_symbol_str(val: Any) -> str:
        """
        Normalize categorical marker symbol for string-based models.

        Rules:
          - None            → ""
          - ""              → ""
          - "nd", "ND", ... → ""
          - "nan", "NaN"... → ""
          - everything else → stripped string
        """
        if val is None:
            return ""
        s = str(val).strip()
        if s.lower() in ("", "nd", "nan"):
            return ""
        return s

    def _build_norm_feature_map(self) -> None:
        """Map normalized feature names to the canonical training feature names."""
        if not self.feature_titles:
            self._norm_feature_map = {}
            return
        mp: Dict[str, str] = {}
        for ft in self.feature_titles:
            key = self._norm_name(ft)
            mp.setdefault(key, ft)  # keep first occurrence on collisions
        self._norm_feature_map = mp

    # -------------------------- training helpers ------------------------------

    def _prepare_X_df(self, X_use: np.ndarray, feature_titles: List[str]) -> pd.DataFrame:
        """
        Convert X_use to DataFrame, normalizing symbols so that missing-like values
        ("", "nd", "nan") become "".
        """
        # Build DataFrame and ensure everything is a string
        X_df = pd.DataFrame(X_use, columns=feature_titles).astype(str)
    
        # Element-wise normalization using column-wise Series.map
        # (avoids both DataFrame.applymap and DataFrame.map)
        for col in X_df.columns:
            X_df[col] = X_df[col].map(self._normalize_symbol_str)
    
        return X_df.astype(str)
        
    # -------------------------- generic accessors -----------------------------

    def get_feature_titles(self) -> List[str] | None:
        """Return the marker names used during training."""
        return self.feature_titles
        
    def get_grouped_feature_values(self, flg_integers: bool = False):
        """
        Return feature values grouped by locus.
    
        If flg_integers=False:
            Simply return self.feature_values as-is (dict expected).
    
        If flg_integers=True:
            Convert values to integers *per locus* and drop values
            that cannot be converted.
    
        Returns
        -------
        dict
            { locus_name : sorted_list_of_values }
            If feature_values was a flat list instead of dict → {"*": [...]}
        """
    
        fv = getattr(self, "feature_values", None)
        if fv is None:
            return {}
    
        # ------------------------------------------------------
        # Case 1 — already a dictionary of lists per locus
        # ------------------------------------------------------
        if isinstance(fv, dict):
            if not flg_integers:
                return fv
    
            grouped = {}
            for locus, vals in fv.items():
                int_list = []
                for v in vals:
                    try:
                        iv = int(str(v).strip())
                        int_list.append(iv)
                    except Exception:
                        # ignore non-numeric values
                        continue
    
                if int_list:
                    grouped[locus] = sorted(set(int_list))
                else:
                    # keep empty lists as empty groups
                    grouped[locus] = []
    
            return grouped
    
        # ------------------------------------------------------
        # Case 2 — flat list of values (rare case)
        # Turn into one group named "*"
        # ------------------------------------------------------
        elif isinstance(fv, (list, tuple, set)):
            if not flg_integers:
                return {"*": list(fv)}
    
            int_list = []
            for v in fv:
                try:
                    iv = int(str(v).strip())
                    int_list.append(iv)
                except Exception:
                    continue
    
            return {"*": sorted(set(int_list))}
    
        # ------------------------------------------------------
        # Unknown structure → attempt best-effort fallback
        # ------------------------------------------------------
        else:
            if not flg_integers:
                return {"*": list(fv)} if isinstance(fv, (list, tuple, set)) else {"*": [fv]}
    
            int_list = []
            try:
                for v in fv:
                    try:
                        iv = int(str(v).strip())
                        int_list.append(iv)
                    except Exception:
                        continue
                return {"*": sorted(set(int_list))}
            except Exception:
                return {}
            

    def get_all_feature_values(self, flg_integers: bool = False):
        """
        Combine all unique values across self.feature_values into one list.
    
        Parameters
        ----------
        flg_integers : bool
            If True:
                - attempt to convert each value to int
                - discard values that cannot be converted (e.g. 'nd', '', None)
    
        Returns
        -------
        list
            A sorted list of unique values.
        """
    
        fv = getattr(self, "feature_values", None)
        if fv is None:
            return []
    
        all_vals = set()
    
        # ------------------------------------------------------
        # CASE 1: dictionary of lists {'marker': [values...]}
        # ------------------------------------------------------
        if isinstance(fv, dict):
            for vals in fv.values():
                for v in vals:
                    all_vals.add(v)
    
        # ------------------------------------------------------
        # CASE 2: flat list of values
        # ------------------------------------------------------
        elif isinstance(fv, (list, tuple, set)):
            for v in fv:
                all_vals.add(v)
    
        else:
            # Unknown format → best effort fallback
            try:
                return sorted(list(set(fv)))
            except Exception:
                return []
    
        # ------------------------------------------------------
        # If integers requested, attempt numeric conversion
        # ------------------------------------------------------
        if flg_integers:
            int_vals = set()
            for v in all_vals:
                try:
                    # Convert only if v represents an integer
                    iv = int(str(v).strip())
                    int_vals.add(iv)
                except Exception:
                    # silently skip values that cannot convert
                    continue
            return sorted(int_vals)
    
        # ------------------------------------------------------
        # Otherwise return as strings (sorted lexicographically)
        # ------------------------------------------------------
        return sorted(map(str, all_vals))
    
    # -------------------------- generic identify ------------------------------

    def identify(
        self,
        markers: Any
    ) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Generic identify() for categorical models using OneHotEncoder-based pipelines
        (plain marker style by default).

        If self.marker_style == "combined":
            - Delegates to self.identify_combined(markers), which expects
              one-hot style markers (e.g. 'marker=value' features).
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.identify_combined(markers)

        # =========================
        # PLAIN MARKER STYLE IDENT
        # =========================
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        if not self.feature_titles:
            raise RuntimeError(
                "Feature titles are not available; ensure train(...) was called with feature_titles."
            )
        if not self.titles:
            raise RuntimeError(
                "Titles mapping is missing; ensure train(...) was called with class labels."
            )

        if self._norm_feature_map is None:
            self._build_norm_feature_map()

        # Prepare a row with defaults = "" for every trained feature (unknown symbol)
        row: Dict[str, str] = {ft: "" for ft in self.feature_titles}

        # Fill from markers with robust name matching
        filled = 0

        def _iter_items(m: Any):
            if hasattr(m, "items"):
                return list(m.items())
            try:
                return list(m)
            except Exception:
                return []

        for k, v in _iter_items(markers):
            norm_k = self._norm_name(k)
            if norm_k in self._norm_feature_map:
                real_name = self._norm_feature_map[norm_k]
                row[real_name] = self._normalize_symbol_str(v)
                filled += 1

        coverage = filled / max(1, len(self.feature_titles))
        if coverage < self.min_coverage_warn:
            msg = (f"⚠️ {self.model}.identify(): low feature coverage: matched {filled} / "
                   f"{len(self.feature_titles)} ({coverage:.1%}). "
                   f"Most columns will be treated as unknown → near-prior probabilities.")
            if self.raise_on_low_coverage:
                raise ValueError(msg)
            else:
                print(msg)

        X_one = pd.DataFrame([row], columns=self.feature_titles).astype(str)

        # Transform + predict probs via the fitted pipeline
        proba = self.predict_proba(X_one)[0]
        titles = (
            self.classes_
            if (getattr(self, "classes_", None) is not None)
            else np.array(self.titles, dtype=object)
        )
        predictions = _proba_to_predictions(proba, titles)
        return self.format_output(predictions)

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE IDENTIFICATION
    # ==================================================
    def identify_combined(
        self,
        markers: Any
    ) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Identification for models trained in 'combined' (one-hot) marker style.

        EXPECTED CONTEXT:
          - self.marker_style == "combined"
          - self.feature_titles are one-hot feature names such as
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', ...
          - The pipeline was trained on numeric one-hot features
            (no internal OneHotEncoder).

        PARAMETERS
        ----------
        markers : Any
            Typically a dict-like containing one-hot-style indicators, e.g.:

                { "yp1108ms45=128": 1, "yp3057ms09=145": 1, ... }

            or the same keys but any truthy value. The caller (e.g.
            identify_records) is expected to build these from a plain
            profile if needed.

        RETURNS
        -------
        dict
            {
              "predictions": [[label, prob], ... sorted by prob desc],
              "odd-ratios": [...],
              "entropy": float
            }
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        if not self.feature_titles:
            raise RuntimeError(
                "Feature titles are not available; ensure train(...) was called with feature_titles."
            )
        if not self.titles:
            raise RuntimeError(
                "Titles mapping is missing; ensure train(...) was called with class labels."
            )

        if self._norm_feature_map is None:
            self._build_norm_feature_map()

        # Row of numeric one-hot features, default = 0.0 (absent)
        row: Dict[str, float] = {ft: 0.0 for ft in self.feature_titles}

        # For coverage calculation in 'combined' mode, we want the fraction of
        # *base markers* (before "=value") that are represented in the input.
        all_base_markers = {ft.split("=", 1)[0] for ft in self.feature_titles}
        seen_base_markers: set[str] = set()

        def _iter_items(m: Any):
            if hasattr(m, "items"):
                return list(m.items())
            try:
                return list(m)
            except Exception:
                return []

        for k, v in _iter_items(markers):
            norm_k = self._norm_name(k)
            if norm_k in self._norm_feature_map:
                real_name = self._norm_feature_map[norm_k]
                # Mark presence (one-hot); ignore actual value magnitude
                row[real_name] = 1.0
                # Track base marker for coverage
                base = real_name.split("=", 1)[0]
                seen_base_markers.add(base)

        coverage = len(seen_base_markers) / max(1, len(all_base_markers))
        if coverage < self.min_coverage_warn:
            msg = (f"⚠️ {self.model}.identify(): low feature coverage (combined mode): "
                   f"matched {len(seen_base_markers)} / {len(all_base_markers)} "
                   f"base markers ({coverage:.1%}). "
                   f"Most columns will be treated as absent → near-prior probabilities.")
            if self.raise_on_low_coverage:
                raise ValueError(msg)
            else:
                print(msg)

        # Single numeric one-hot row
        X_one = pd.DataFrame([row], columns=self.feature_titles)

        # Transform + predict probs via the fitted pipeline
        proba = self.predict_proba(X_one)[0]
        titles = (
            self.classes_
            if (getattr(self, "classes_", None) is not None)
            else np.array(self.titles, dtype=object)
        )
        predictions = _proba_to_predictions(proba, titles)
        return self.format_output(predictions)
    # -------------------------- output formatting -----------------------------

    def format_output(
        self,
        predictions: Dict[Any, float]
    ) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Format predictions dict {label: prob} into:
          - sorted list of [label, prob]
          - simple “odd-ratios” for top vs others
          - normalized entropy of the distribution
        """
        def get_odd_value(val: float) -> float:
            return val / (1 - val) if val != 1 else 10.0

        def get_entropy(values: List[float]) -> float:
            n = len(values)
            if n < 2:
                return 0.0
            if any(float(v) in (0.0, 1.0) for v in values):
                return 0.0
            arr = np.array(values, dtype=float)
            probs = arr / arr.sum()
            return float(-np.sum(probs * np.log2(probs)) / np.log2(n))

        # Sort predictions by probability descending
        items_sorted = sorted(
            [list(item) for item in predictions.items()],
            key=lambda ls: ls[1],
            reverse=True
        )
        values = [item[1] for item in items_sorted]
        best_odd = get_odd_value(values[0])
        odds = [1.0] + [best_odd - get_odd_value(values[i]) for i in range(1, len(values))]
        return {
            "predictions": items_sorted,
            "odd-ratios": odds,
            "entropy": get_entropy(values)
        }

    # -------------------------- generic predict_proba -------------------------

    def predict_proba(self, X_df_or_array: Any) -> np.ndarray:
        """
        Delegates to the classifier inside the pipeline.

        For categorical=True:
          - Pass a pandas DataFrame with raw symbols, using the SAME columns
            as self.feature_titles.

        For categorical=False:
          - You may pass a numeric array already in the encoded feature space.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        clf = self.pipeline.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "predict_proba"):
            raise RuntimeError("Classifier in the pipeline does not support predict_proba().")
        return self.pipeline.predict_proba(X_df_or_array)


# =============================================================================
# MLP Class (Multilayer Perceptron model)
# =============================================================================

class MLP(MLearn):
    """
    Multilayer Perceptron classifier treating each cell as a SYMBOL (categorical state).
    """

    def __init__(
        self,
        marker_style: str = "plain",    # Marker title style: plain (default) like 'marker_1' or combined (one-hot) like 'marker_1=0'
        hidden_layer_sizes: Union[tuple, list] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 1000,
        early_stopping: bool = False,
        random_state: int = 42,
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False
    ) -> None:
        try:
            # Just to ensure sklearn is installed
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="MLP",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.hidden_layer_sizes = (
            tuple(hidden_layer_sizes)
            if isinstance(hidden_layer_sizes, (list, tuple))
            else (hidden_layer_sizes,)
        )
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.early_stopping = bool(early_stopping)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "MLP":
        """
        Dispatching train method for MLP.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. '101361', '366141', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion ('0','1','nan','nd','') is handled internally by
              OneHotEncoder in the pipeline.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # Auto-generate generic names if not provided
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # --- HARD GUARD against 'marker=value' style in plain mode ---
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "MLP.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by MLP in 'plain' mode.\n"
                "Please pass ONE column per original marker instead "
                "(e.g. '101361', '366141', ...), and let OneHotEncoder "
                "handle category expansion internally."
            )

        # Store clean marker titles, which MUST match what identify() will use
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of normalized symbols ---
        X_df = self._prepare_X_df(X_use, self.feature_titles)


        # --- Collect unique values per marker (no duplicates) ---
        # X_df is already normalized ("" / nd / nan -> ""), so we take it as ground truth.
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # --- Preprocessor ---
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # --- MLP classifier ---
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )

        # --- Build pipeline ---
        self.pipeline = _P([("pre", pre), ("clf", mlp)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)

        # Build normalized-name map for robust identify()
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "MLP":
        """
        Train an MLP in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (e.g. from pd.get_dummies)
          - feature_titles are 'marker=value' names, e.g.:
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', 'yp1108ms45=nan', ...
          - No additional OneHotEncoder is applied; features are used as-is.

        This mode is for backward compatibility with older 'combined' models.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # Auto-generate generic names if not provided
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In 'combined' mode we EXPECT many 'marker=value' titles,
        # so we do NOT raise an error if '=' is present.
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of encoded features (numeric one-hot) ---
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)


        # --- Collect unique values per marker from one-hot representation ---
        feature_values: dict[str, list[str]] = {}

        for col_idx, full_name in enumerate(self.feature_titles):
            # Skip values that never occur in the training data (column all zeros)
            if not np.any(X_df.iloc[:, col_idx].values):
                continue

            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                # fallback: treat full_name as marker, with implicit value "1"
                marker, value = full_name, "1"

            feature_values.setdefault(marker, []).append(str(value))

        # Deduplicate + sort
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # --- No extra preprocessing: data is already encoded ---
        pre = "passthrough"

        # --- MLP classifier ---
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )

        # --- Build pipeline ---
        self.pipeline = _P([("pre", pre), ("clf", mlp)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)

        # Build normalized-name map for robust identify()
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

# =============================================================================
# RF Class (Random Forest)
# =============================================================================

class RF(MLearn):
    """
    Random Forest classifier treating each cell as a SYMBOL (categorical state).
    """

    def __init__(
        self,
        marker_style: str = "plain",    # Marker title style: plain (default) like 'marker_1' or combined (one-hot) like 'marker_1=0'
        n_estimators: int = 300,
        max_depth: Union[int, None] = None,
        max_features: Union[str, int, float, None] = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        class_weight: Union[str, dict, None] = None,
        oob_score: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False
    ) -> None:
        try:
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="RF",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "RF":
        """
        Dispatching train method for Random Forest.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. '101361', '366141', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion ('0','1','nan','nd','') is handled internally by
              OneHotEncoder in the pipeline.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # HARD GUARD against 'marker=value' style in plain mode
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "RF.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by RF in 'plain' mode.\n"
                "Please pass ONE column per original marker instead."
            )

        # Store clean marker titles, which MUST match what identify() will use
        self.feature_titles = list(feature_titles)

        # Labels, first-seen order preserved
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singleton classes
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # DataFrame with normalized symbols
        X_df = self._prepare_X_df(X_use, self.feature_titles)


        # --- Collect unique values per marker (no duplicates) ---
        # X_df is already normalized ("" / nd / nan -> ""), so we take it as ground truth.
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # Train/test split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # Preprocessor: internal OneHotEncoder if categorical
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # RF classifier
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # Build pipeline: [preprocessor] -> RF
        self.pipeline = _P([("pre", pre), ("clf", rf)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "RF":
        """
        Train a Random Forest in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (e.g. from pd.get_dummies)
          - feature_titles are 'marker=value' names, e.g.:
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', 'yp1108ms45=nan', ...
          - No additional OneHotEncoder is applied; features are used as-is.

        This mode is for backward compatibility with older 'combined' models.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In 'combined' mode we EXPECT many 'marker=value' titles,
        # so we do NOT raise an error if '=' is present.
        self.feature_titles = list(feature_titles)

        # Labels
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singletons
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # DataFrame of encoded features (numeric one-hot)
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)


        # --- Collect unique values per marker from one-hot representation ---
        feature_values: dict[str, list[str]] = {}

        for col_idx, full_name in enumerate(self.feature_titles):
            # Skip values that never occur in the training data (column all zeros)
            if not np.any(X_df.iloc[:, col_idx].values):
                continue

            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                # fallback: treat full_name as marker, with implicit value "1"
                marker, value = full_name, "1"

            feature_values.setdefault(marker, []).append(str(value))

        # Deduplicate + sort
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # No extra preprocessing: data is already encoded
        pre = "passthrough"

        # RF classifier
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.pipeline = _P([("pre", pre), ("clf", rf)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

# =============================================================================
# XGBoost Class (Extreme Gradient Boosting)
# =============================================================================

class XGBoost(MLearn):
    """
    Extreme Gradient Boosting classifier (xgboost.XGBClassifier) for categorical marker matrices.

    Same interface as RF/KNN:
      - train(X, y, feature_titles=None, test_size=0.2, drop_singletons=True) -> self
      - identify(markers) (inherited from MLearn; supports plain + combined)
    """

    def __init__(
        self,
        marker_style: str = "plain",   # "plain" (default) or "combined"
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        # optional: imbalance handling (binary or multiclass via sample_weight)
        scale_pos_weight: float | None = None,
        # evaluation split / reproducibility
        random_state: int = 42,
        n_jobs: int = -1,
        # keep same knobs as base
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False,
        # xgboost verbosity
        verbosity: int = 0
    ) -> None:
        # NOTE: xgboost is not part of sklearn; it must be installed separately.
        try:
            import xgboost  # noqa
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline  # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            missing = e.name
            print(f"❌ Missing required module: {missing}")
            if missing == "xgboost":
                print("➡️  Please install XGBoost using: pip install xgboost")
            else:
                print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="XGBoost",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.min_child_weight = float(min_child_weight)
        self.gamma = float(gamma)
        self.scale_pos_weight = scale_pos_weight if scale_pos_weight is None else float(scale_pos_weight)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.verbosity = int(verbosity)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "XGBoost":
        """
        Dispatching train method for XGBoost.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
            - Delegates to self.train_combined(...)

        Otherwise ("plain"):
            - X has one column per marker, raw categorical symbols
            - OneHotEncoder is applied internally when categorical=True
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        from xgboost import XGBClassifier

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # HARD GUARD against 'marker=value' style in plain mode
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "XGBoost.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by XGBoost in 'plain' mode.\n"
                "Please pass ONE column per original marker instead."
            )

        # Store marker titles for identify()
        self.feature_titles = list(feature_titles)

        # Labels, first-seen order preserved
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singleton classes
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # DataFrame with normalized symbols
        X_df = self._prepare_X_df(X_use, self.feature_titles)

        # Collect unique values per marker (already normalized)
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # Preprocessor
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # Multiclass objective selection
        n_classes = int(len(set(y_train)))
        if n_classes <= 1:
            raise ValueError("XGBoost.train(): need at least 2 classes to train.")

        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        # NOTE: XGBClassifier uses its own random_state/seed.
        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
        )

        # Optional imbalance handling (binary only is typical)
        if (self.scale_pos_weight is not None) and (n_classes == 2):
            xgb.set_params(scale_pos_weight=self.scale_pos_weight)

        # Pipeline
        self.pipeline = _P([("pre", pre), ("clf", xgb)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "XGBoost":
        """
        Train XGBoost in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (numeric)
          - feature_titles are 'marker=value' names
          - No extra OneHotEncoder; features are used as-is.
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P
        from xgboost import XGBClassifier

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        self.feature_titles = list(feature_titles)

        # Labels
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singletons
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        X_df = pd.DataFrame(X_use, columns=self.feature_titles)

        # Collect values per base marker from one-hot titles (only columns that occur)
        feature_values: dict[str, list[str]] = {}
        for col_idx, full_name in enumerate(self.feature_titles):
            if not np.any(X_df.iloc[:, col_idx].values):
                continue
            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                marker, value = full_name, "1"
            feature_values.setdefault(marker, []).append(str(value))
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # Objective selection
        n_classes = int(len(set(y_train)))
        if n_classes <= 1:
            raise ValueError("XGBoost.train_combined(): need at least 2 classes to train.")

        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
        )

        if (self.scale_pos_weight is not None) and (n_classes == 2):
            xgb.set_params(scale_pos_weight=self.scale_pos_weight)

        self.pipeline = _P([("pre", "passthrough"), ("clf", xgb)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

# =============================================================================
# NBayes Class (Naive Bayes)
# =============================================================================

class NBayes(MLearn):
    """
    Naive Bayes classifier for categorical marker matrices.

    Same interface as RF/KNN/XGBoost:
      - train(X, y, feature_titles=None, test_size=0.2, drop_singletons=True) -> self
      - identify(markers) (inherited from MLearn; supports plain + combined)

    Implementation:
      - marker_style="plain": uses sklearn.naive_bayes.CategoricalNB with per-column ordinal encoding
      - marker_style="combined": uses sklearn.naive_bayes.MultinomialNB on one-hot numeric features
    """

    def __init__(
        self,
        marker_style: str = "plain",   # "plain" (default) or "combined"
        alpha: float = 1.0,           # smoothing (CategoricalNB & MultinomialNB)
        fit_prior: bool = True,
        class_prior: list[float] | None = None,
        categorical: bool = True,     # kept for interface consistency; NB here is categorical by design
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False,
        random_state: int = 42
    ) -> None:
        try:
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="NBayes",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.class_prior = class_prior
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "NBayes":
        """
        Dispatching train method for Naive Bayes.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
            - Delegates to self.train_combined(...)

        Otherwise ("plain"):
            - X has one column per marker, raw categorical symbols
            - Uses OrdinalEncoder -> CategoricalNB
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.naive_bayes import CategoricalNB
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # HARD GUARD against 'marker=value' style in plain mode
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "NBayes.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by NBayes in 'plain' mode.\n"
                "Please pass ONE column per original marker instead."
            )

        # Store marker titles for identify()
        self.feature_titles = list(feature_titles)

        # Labels, first-seen order preserved
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singleton classes
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # DataFrame with normalized symbols
        X_df = self._prepare_X_df(X_use, self.feature_titles)

        # Collect unique values per marker (already normalized)
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # Ordinal encoding per column, unknown -> -1 (safe for predict time)
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        nb = CategoricalNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior
        )

        self.pipeline = _P([("pre", enc), ("clf", nb)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "NBayes":
        """
        Train Naive Bayes in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (numeric)
          - feature_titles are 'marker=value' names
          - Uses MultinomialNB (works well with sparse/binary counts)
        """
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In combined mode '=' is expected
        self.feature_titles = list(feature_titles)

        # Labels
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singletons
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        X_df = pd.DataFrame(X_use, columns=self.feature_titles)

        # Collect values per base marker from one-hot titles (only columns that occur)
        feature_values: dict[str, list[str]] = {}
        for col_idx, full_name in enumerate(self.feature_titles):
            if not np.any(X_df.iloc[:, col_idx].values):
                continue
            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                marker, value = full_name, "1"
            feature_values.setdefault(marker, []).append(str(value))
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # MultinomialNB expects non-negative features (one-hot is fine)
        nb = MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior
        )

        self.pipeline = _P([("pre", "passthrough"), ("clf", nb)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self


# =============================================================================
# KNN Class (k-Nearest Neighbours)
# =============================================================================

class KNN(MLearn):
    """
    k-Nearest Neighbours classifier for categorical marker matrices.

    Supports the same interface as RF:
      - train(X, y, feature_titles=None, test_size=0.2, drop_singletons=True) -> self
      - identify(markers)  (inherited from MLearn; supports plain + combined)
    """

    def __init__(
        self,
        marker_style: str = "plain",     # "plain" (default) or "combined"
        n_neighbors: int = 5,
        weights: str = "distance",      # "uniform" or "distance"
        metric: str = "minkowski",      # KNN metric in encoded space
        p: int = 2,                     # for minkowski (p=2 -> euclidean)
        n_jobs: int = -1,
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False,
        random_state: int = 42
    ) -> None:
        try:
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="KNN",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.metric = str(metric)
        self.p = int(p)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "KNN":
        """
        Dispatching train method for KNN.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. 'yp1108ms45', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion is handled internally by OneHotEncoder in the pipeline
              when categorical=True.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # HARD GUARD against 'marker=value' style in plain mode
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "KNN.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by KNN in 'plain' mode.\n"
                "Please pass ONE column per original marker instead."
            )

        # Store clean marker titles for identify()
        self.feature_titles = list(feature_titles)

        # Labels, first-seen order preserved
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singleton classes
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # DataFrame with normalized symbols
        X_df = self._prepare_X_df(X_use, self.feature_titles)

        # Collect unique values per marker (already normalized)
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # Preprocessor
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # KNN classifier
        knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )

        # Pipeline
        self.pipeline = _P([("pre", pre), ("clf", knn)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "KNN":
        """
        Train KNN in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (numeric)
          - feature_titles are 'marker=value' names
          - No additional OneHotEncoder is applied; features are used as-is.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # Validate feature length
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In combined mode '=' is expected
        self.feature_titles = list(feature_titles)

        # Labels
        self.titles, y_idx = _order_preserving_label_map(y)

        # Drop singletons
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # DataFrame of encoded features (numeric one-hot)
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)

        # Collect values per base marker from one-hot titles (only columns that occur)
        feature_values: dict[str, list[str]] = {}
        for col_idx, full_name in enumerate(self.feature_titles):
            if not np.any(X_df.iloc[:, col_idx].values):
                continue
            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                marker, value = full_name, "1"
            feature_values.setdefault(marker, []).append(str(value))

        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # Split
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        pre = "passthrough"

        knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )

        self.pipeline = _P([("pre", pre), ("clf", knn)])

        # Fit & evaluate
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self


# =============================================================================
# SVC Class (Support Vector Machine)
# =============================================================================

class SVC(MLearn):
    """
    Support Vector Machine classifier treating each cell as a SYMBOL (categorical state).
    """

    def __init__(
        self,
        marker_style: str = "plain",    # Marker title style: plain (default) like 'marker_1' or combined (one-hot) like 'marker_1=0'
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        class_weight: Union[str, dict, None] = None,
        random_state: int = 42,
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False
    ) -> None:
        try:
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="SVC",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.class_weight = class_weight
        self.random_state = random_state

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "SVC":
        """
        Dispatching train method for SVC.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. '101361', '366141', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion ('0','1','nan','nd','') is handled internally by
              OneHotEncoder in the pipeline.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.svm import SVC as _SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # --- HARD GUARD against 'marker=value' style in plain mode ---
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "SVC.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by SVC in 'plain' mode.\n"
                "Please pass ONE column per original marker instead "
                "(e.g. '101361', '366141', ...), and let OneHotEncoder "
                "handle category expansion internally."
            )

        # Store clean marker titles, which MUST match what identify() will use
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of normalized symbols ---
        X_df = self._prepare_X_df(X_use, self.feature_titles)


        # --- Collect unique values per marker (no duplicates) ---
        # X_df is already normalized ("" / nd / nan -> ""), so we take it as ground truth.
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # --- Preprocessor ---
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # --- SVM classifier ---
        svm = _SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_state,
        )

        # --- Build pipeline ---
        self.pipeline = _P([("pre", pre), ("clf", svm)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "SVC":
        """
        Train an SVM (SVC) in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (e.g. from pd.get_dummies)
          - feature_titles are 'marker=value' names, e.g.:
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', 'yp1108ms45=nan', ...
          - No additional OneHotEncoder is applied; features are used as-is.

        This mode is for backward compatibility with older 'combined' models.
        """
        from sklearn.svm import SVC as _SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In 'combined' mode we EXPECT many 'marker=value' titles,
        # so we do NOT raise an error if '=' is present.
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of encoded features (numeric one-hot) ---
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)


        # --- Collect unique values per marker from one-hot representation ---
        feature_values: dict[str, list[str]] = {}

        for col_idx, full_name in enumerate(self.feature_titles):
            # Skip values that never occur in the training data (column all zeros)
            if not np.any(X_df.iloc[:, col_idx].values):
                continue

            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                # fallback: treat full_name as marker, with implicit value "1"
                marker, value = full_name, "1"

            feature_values.setdefault(marker, []).append(str(value))

        # Deduplicate + sort
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # --- No extra preprocessing: data is already encoded ---
        pre = "passthrough"

        # --- SVM classifier ---
        svm = _SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_state,
        )

        # --- Build pipeline ---
        self.pipeline = _P([("pre", pre), ("clf", svm)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self
        
# =============================================================================
# LR Class (Logistic Regression)
# =============================================================================

class LR(MLearn):
    """
    Logistic Regression classifier treating each cell as a SYMBOL (categorical state).
    """

    def __init__(
        self,
        marker_style: str = 'plain',
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Union[str, dict, None] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False
    ) -> None:
        try:
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="LR",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.C = float(C)
        self.penalty = penalty
        self.solver = solver
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "LR":
        """
        Dispatching train method for Logistic Regression.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. '101361', '366141', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion ('0','1','nan','nd','') is handled internally by
              OneHotEncoder in the pipeline.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # --- HARD GUARD against 'marker=value' style in plain mode ---
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "LR.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by LR in 'plain' mode.\n"
                "Please pass ONE column per original marker instead "
                "(e.g. '101361', '366141', ...), and let OneHotEncoder "
                "handle category expansion internally."
            )

        # Store clean marker titles, which MUST match what identify() will use
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of RAW symbols, with ""/nd/nan normalized to "" ---
        # Previously: X_df = self._prepare_X_df(X_use, self.feature_titles)
        # Now do the normalization here using DataFrame.map (no applymap).
        X_df = pd.DataFrame(X_use, columns=self.feature_titles).astype(str)
        # normalize each cell ("" / "nd" / "nan" → "")
        X_df = X_df.map(self._normalize_symbol_str)

        # --- Collect unique values per marker (no duplicates) ---
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # --- Preprocessor: OneHotEncode all marker columns if categorical=True ---
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # --- Logistic Regression classifier ---
        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # --- Build pipeline: [preprocessor] -> LR ---
        self.pipeline = _P([("pre", pre), ("clf", lr)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True
    ) -> "LR":
        """
        Train Logistic Regression in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (e.g. from pd.get_dummies)
          - feature_titles are 'marker=value' names, e.g.:
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', 'yp1108ms45=nan', ...
          - No additional OneHotEncoder is applied; features are used as-is.

        This mode is for backward compatibility with older 'combined' models.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In 'combined' mode we EXPECT many 'marker=value' titles,
        # so we do NOT raise an error if '=' is present.
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of encoded features (typically 0/1) ---
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)

        # --- Collect unique values per marker from one-hot representation ---
        feature_values: dict[str, list[str]] = {}

        for col_idx, full_name in enumerate(self.feature_titles):
            # Skip values that never occur in the training data (column all zeros)
            if not np.any(X_df.iloc[:, col_idx].values):
                continue

            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                # fallback: treat full_name as marker, with implicit value "1"
                marker, value = full_name, "1"

            feature_values.setdefault(marker, []).append(str(value))

        # Deduplicate + sort
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # --- No extra preprocessing: data is already encoded ---
        pre = "passthrough"

        # --- Logistic Regression classifier ---
        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # --- Build pipeline: [passthrough] -> LR ---
        self.pipeline = _P([("pre", pre), ("clf", lr)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        return self

# =============================================================================
# MBCS Class (Matrix-Based Classifier Search)
# =============================================================================

class MBCS(MLearn):
    """
    Matrix-Based Classifier Search (MBCS)

    PURPOSE:
      - Given a feature matrix X (typically already one-hot encoded) and labels y,
        evaluate several candidate classifiers via cross-validation.
      - Suggest a "recommended" classifier for this dataset.
      - Provide a short textual rationale and a crude "probe" feature-importance
        profile via a RandomForest probe.

    INTERFACE:
      - train(X, y)    -> fits the MBCS analyzer (no feature titles needed)
      - identify()     -> returns recommendation + summary dict.
    """

    def __init__(
        self,
        marker_style: str = "plain",    # Marker title style: plain (default) like 'marker_1' or combined (one-hot) like 'marker_1=0'
        random_state: int = 42,
        cv_folds: int = 5,
        test_size: float = 0.2,
        n_jobs: int = -1
    ) -> None:
        try:
            from sklearn.model_selection import train_test_split, cross_val_score # noqa
            from sklearn.linear_model import LogisticRegression                   # noqa
            from sklearn.ensemble import RandomForestClassifier                   # noqa
            from sklearn.naive_bayes import GaussianNB                            # noqa
            from sklearn.svm import SVC                                           # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        # MBCS does not use categorical/identify machinery, but we still reuse the base shell.
        super().__init__(model="MBCS", marker_style=marker_style, categorical=False)

        self.random_state = int(random_state)
        self.cv_folds = int(cv_folds)
        self.test_size = float(test_size)
        self.n_jobs = int(n_jobs)

        # Stored after training
        self.X_shape: Tuple[int, int] | None = None
        self.y_classes_: np.ndarray | None = None
        self.class_counts_: Dict[Any, int] | None = None
        self.cv_results_: Dict[str, Dict[str, float]] | None = None
        self.recommended_: str | None = None
        self.probe_scores_: List[float] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> "MBCS":
        """
        Run matrix-based classifier search.

        Dispatches by self.marker_style:

        - If self.marker_style == "combined" (legacy/one-hot style):
            X is assumed to be already numeric and suitable for sklearn
            (typically one-hot encoded, e.g. via pd.get_dummies).
            Delegates to self.train_combined(X, y).

        - Otherwise (self.marker_style == "plain"):
            X may contain raw categorical symbols. We internally
            one-hot encode them (pd.get_dummies) and then run the
            same MBCS search pipeline.
        """
        style = getattr(self, "marker_style", "combined")
        if style == "combined":
            return self.train_combined(X, y)

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC as _SVC

        # Convert to array first
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(f"MBCS.train(): X must be 2D, got shape {X_arr.shape}")
        if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"MBCS.train(): y must be 1D of length {X_arr.shape[0]}, got shape {y_arr.shape}"
            )

        # ----------------------------------------------------
        # Collect all unique values present in the training X
        # (raw symbols if non-numeric, numeric values otherwise)
        # ----------------------------------------------------
        try:
            if not np.issubdtype(X_arr.dtype, np.number):
                # Work on raw categorical symbols (before one-hot)
                df_raw = pd.DataFrame(X_arr)
                all_vals: set[str] = set()
                for col in df_raw.columns:
                    all_vals.update(df_raw[col].astype(str).unique().tolist())
                self.feature_values = sorted(all_vals)
            else:
                # Numeric matrix: unique finite numeric values
                flat = X_arr.ravel()
                if np.issubdtype(flat.dtype, np.floating):
                    flat = flat[~np.isnan(flat)]
                self.feature_values = sorted(map(float, np.unique(flat)))
        except Exception as e:
            print(f"⚠️  MBCS: failed to collect feature_values in plain mode: {e}")
            self.feature_values = None

        # If X is not numeric, one-hot encode it
        if not np.issubdtype(X_arr.dtype, np.number):
            df = pd.DataFrame(X_arr)
            # Optional: cast to 'category' for clarity
            for c in df.columns:
                df[c] = df[c].astype("category")
            X_arr = pd.get_dummies(df, prefix_sep="=", dummy_na=True).values.astype(float)

        n_samples, n_features = X_arr.shape
        classes, counts = np.unique(y_arr, return_counts=True)
        self.X_shape = (n_samples, n_features)
        self.y_classes_ = classes
        self.class_counts_ = {cls: int(cnt) for cls, cnt in zip(classes, counts, strict=False)}

        candidates = {
            "LogisticRegression": LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                n_jobs=self.n_jobs,
                class_weight=None,
                multi_class="auto",
                random_state=self.random_state,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                oob_score=False,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "GaussianNB": GaussianNB(),
            "SVC_RBF": _SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=False,
                random_state=self.random_state,
            ),
        }

        cv_results: Dict[str, Dict[str, float]] = {}

        for name, clf in candidates.items():
            try:
                scores = cross_val_score(
                    clf,
                    X_arr,
                    y_arr,
                    cv=self.cv_folds,
                    scoring="accuracy",
                    n_jobs=self.n_jobs,
                )
                cv_results[name] = {
                    "mean_accuracy": float(scores.mean()),
                    "std_accuracy": float(scores.std()),
                }
            except Exception as e:
                print(f"⚠️  MBCS: classifier {name} failed during CV: {e}")
                cv_results[name] = {
                    "mean_accuracy": float("nan"),
                    "std_accuracy": float("nan"),
                }

        self.cv_results_ = cv_results

        best_name = None
        best_score = -np.inf
        for name, res in cv_results.items():
            score = res.get("mean_accuracy", float("nan"))
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_name = name
        self.recommended_ = best_name

        # Probe feature importance
        self.probe_scores_ = None
        try:
            rf_probe = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                oob_score=False,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            rf_probe.fit(X_arr, y_arr)
            if hasattr(rf_probe, "feature_importances_"):
                self.probe_scores_ = list(map(float, rf_probe.feature_importances_))
        except Exception as e:
            print(f"⚠️  MBCS: RandomForest probe failed: {e}")
            self.probe_scores_ = None

        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(self, X: np.ndarray, y: np.ndarray) -> "MBCS":
        """
        Run matrix-based classifier search in 'combined' (one-hot) style.

        EXPECTED INPUT:
          - X is already numeric (typically one-hot encoded,
            e.g. from pd.get_dummies on a categorical matrix).
          - No further encoding is performed here.

        Behaviour matches the original MBCS.train implementation.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC as _SVC

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(f"MBCS.train_combined(): X must be 2D, got shape {X_arr.shape}")
        if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"MBCS.train_combined(): y must be 1D of length {X_arr.shape[0]}, got shape {y_arr.shape}"
            )

        # ----------------------------------------------------
        # Collect all unique values present in the one-hot X
        # (typically [0.0, 1.0])
        # ----------------------------------------------------
        try:
            flat = X_arr.ravel()
            if np.issubdtype(flat.dtype, np.floating):
                flat = flat[~np.isnan(flat)]
            self.feature_values = sorted(map(float, np.unique(flat)))
        except Exception as e:
            print(f"⚠️  MBCS: failed to collect feature_values in combined mode: {e}")
            self.feature_values = None

        n_samples, n_features = X_arr.shape
        classes, counts = np.unique(y_arr, return_counts=True)
        self.X_shape = (n_samples, n_features)
        self.y_classes_ = classes
        self.class_counts_ = {cls: int(cnt) for cls, cnt in zip(classes, counts, strict=False)}

        candidates = {
            "LogisticRegression": LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                n_jobs=self.n_jobs,
                class_weight=None,
                multi_class="auto",
                random_state=self.random_state,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                oob_score=False,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "GaussianNB": GaussianNB(),
            "SVC_RBF": _SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=False,
                random_state=self.random_state,
            ),
        }

        cv_results: Dict[str, Dict[str, float]] = {}

        for name, clf in candidates.items():
            try:
                scores = cross_val_score(
                    clf,
                    X_arr,
                    y_arr,
                    cv=self.cv_folds,
                    scoring="accuracy",
                    n_jobs=self.n_jobs,
                )
                cv_results[name] = {
                    "mean_accuracy": float(scores.mean()),
                    "std_accuracy": float(scores.std()),
                }
            except Exception as e:
                print(f"⚠️  MBCS: classifier {name} failed during CV (combined): {e}")
                cv_results[name] = {
                    "mean_accuracy": float("nan"),
                    "std_accuracy": float("nan"),
                }

        self.cv_results_ = cv_results

        best_name = None
        best_score = -np.inf
        for name, res in cv_results.items():
            score = res.get("mean_accuracy", float("nan"))
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_name = name
        self.recommended_ = best_name

        # Probe feature importance
        self.probe_scores_ = None
        try:
            rf_probe = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                oob_score=False,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            rf_probe.fit(X_arr, y_arr)
            if hasattr(rf_probe, "feature_importances_"):
                self.probe_scores_ = list(map(float, rf_probe.feature_importances_))
        except Exception as e:
            print(f"⚠️  MBCS: RandomForest probe failed (combined): {e}")
            self.probe_scores_ = None

        return self

    # -------------------------- generic identify ------------------------------

    def identify(self) -> Dict[str, Any]:
        """
        Return the MBCS recommendation and summary.

        Dispatches by self.marker_style:

        - If self.marker_style == "combined":
            use identify_combined() (intended for models trained on
            one-hot encoded matrices, e.g. pd.get_dummies output).

        - Otherwise (e.g. self.marker_style == "plain"):
            use the standard identification based on the stored
            dataset / CV results.
        """
        style = getattr(self, "marker_style", "combined")
        if style == "combined":
            return self.identify_combined()

        # =========================
        # PLAIN MATRIX STYLE IDENT
        # =========================
        if self.X_shape is None or self.y_classes_ is None or self.cv_results_ is None:
            raise RuntimeError("MBCS.identify(): model not trained yet. Call train(X, y) first.")

        n_samples, n_features = self.X_shape
        n_classes = len(self.y_classes_)

        dataset_summary = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
            "class_counts": self.class_counts_,
        }

        rationale: List[str] = []

        # Class imbalance rationale
        if self.class_counts_:
            max_cnt = max(self.class_counts_.values())
            min_cnt = min(self.class_counts_.values())
            if max_cnt > 0:
                imbalance = (max_cnt - min_cnt) / max_cnt
            else:
                imbalance = 0.0
            rationale.append(
                f"Class imbalance (max-min)/max ≈ {imbalance:.2f} "
                f"with counts: {self.class_counts_}"
            )

        # Cross-validation summary
        if self.cv_results_:
            parts = []
            for name, res in self.cv_results_.items():
                mean_acc = res.get("mean_accuracy", float("nan"))
                std_acc = res.get("std_accuracy", float("nan"))
                if np.isnan(mean_acc):
                    parts.append(f"{name}: failed")
                else:
                    parts.append(f"{name}: {mean_acc:.3f} ± {std_acc:.3f}")
            rationale.append("Cross-validation accuracies → " + "; ".join(parts))

        # Recommendation
        if self.recommended_ is not None:
            rationale.append(
                f"Recommended classifier based on mean CV accuracy: {self.recommended_}"
            )
        else:
            rationale.append("No reliable recommendation (all CV scores failed or NaN).")

        return {
            "recommendation": self.recommended_,
            "rationale": rationale,
            "dataset_summary": dataset_summary,
            "probe_scores": self.probe_scores_,
        }

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE IDENTIFICATION
    # ==================================================
    def identify_combined(self) -> Dict[str, Any]:
        """
        MBCS identification for 'combined' (one-hot) matrix style.

        NOTE:
          - For MBCS, 'plain' vs 'combined' refers to how X was encoded at train time
            (raw vs one-hot), but identification is always based on the stored
            dataset summary and CV results, not on per-record markers.
          - Therefore, this method returns the same type of summary as identify().

        Returns a dict with keys:
          - "recommendation": str or None
          - "rationale": List[str]
          - "dataset_summary": dict
          - "probe_scores": List[float] or None
        """
        if self.X_shape is None or self.y_classes_ is None or self.cv_results_ is None:
            raise RuntimeError(
                "MBCS.identify_combined(): model not trained yet. Call train(X, y) first."
            )

        n_samples, n_features = self.X_shape
        n_classes = len(self.y_classes_)

        dataset_summary = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
            "class_counts": self.class_counts_,
        }

        rationale: List[str] = []

        # Class imbalance rationale
        if self.class_counts_:
            max_cnt = max(self.class_counts_.values())
            min_cnt = min(self.class_counts_.values())
            if max_cnt > 0:
                imbalance = (max_cnt - min_cnt) / max_cnt
            else:
                imbalance = 0.0
            rationale.append(
                f"Class imbalance (max-min)/max ≈ {imbalance:.2f} "
                f"with counts: {self.class_counts_}"
            )

        # Cross-validation summary
        if self.cv_results_:
            parts = []
            for name, res in self.cv_results_.items():
                mean_acc = res.get("mean_accuracy", float("nan"))
                std_acc = res.get("std_accuracy", float("nan"))
                if np.isnan(mean_acc):
                    parts.append(f"{name}: failed")
                else:
                    parts.append(f"{name}: {mean_acc:.3f} ± {std_acc:.3f}")
            rationale.append("Cross-validation accuracies → " + "; ".join(parts))

        # Recommendation
        if self.recommended_ is not None:
            rationale.append(
                f"Recommended classifier based on mean CV accuracy: {self.recommended_}"
            )
        else:
            rationale.append("No reliable recommendation (all CV scores failed or NaN).")

        return {
            "recommendation": self.recommended_,
            "rationale": rationale,
            "dataset_summary": dataset_summary,
            "probe_scores": self.probe_scores_,
        }
# =============================================================================
# DT Class (Decision Tree)
# =============================================================================

class DT(MLearn):
    """
    Decision Tree classifier treating each cell as a SYMBOL (categorical state).

    DESIGN:
      - X passed to `train()` must have ONE column per original marker (e.g. '101361').
      - `feature_titles` must be those marker names, NOT 'marker=value' names.
      - Category expansion (0/1/NaN/""/nd → one-hot) is handled internally by OneHotEncoder.
      - `identify()` accepts a dict like {"101361": 0, "366141": 1, ...}
        using the SAME marker names as in `feature_titles`.
    """

    def __init__(self,
                 marker_style: str = 'plain',
                 criterion: str = "gini",
                 splitter: str = "best",
                 max_depth: Union[int, None] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float, None] = None,
                 class_weight: Union[dict, list, str, None] = None,
                 random_state: int = 42,
                 categorical: bool = True,
                 min_coverage_warn: float = 0.2,
                 raise_on_low_coverage: bool = False) -> None:
        try:
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
            from sklearn.tree import DecisionTreeClassifier       # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="DT",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage
        )

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_titles: List[str] | None = None,
              test_size: float = 0.2,
              drop_singletons: bool = True) -> "DT":
        """
        Dispatching train method.

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. '101361', '366141', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion ('0','1','nan','nd','') is handled internally by
              OneHotEncoder in the pipeline.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )

        # =========================
        # PLAIN MARKER STYLE TRAIN
        # =========================
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
            from sklearn.pipeline import Pipeline as _P
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # Auto-generate generic names if not provided
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # --- HARD GUARD against 'marker=value' style in plain mode ---
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "DT.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported by DT in 'plain' mode.\n"
                "Please pass ONE column per original marker instead "
                "(e.g. '101361', '366141', ...), and let OneHotEncoder "
                "handle category expansion internally."
            )

        # Store clean marker titles, which MUST match what identify() will use
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of RAW symbols, with ""/nd/nan normalized to "" ---
        X_df = self._prepare_X_df(X_use, self.feature_titles)


        # --- Collect unique values per marker (no duplicates) ---
        # X_df is already normalized ("" / nd / nan -> ""), so we take it as ground truth.
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # --- Preprocessor: OneHotEncode all marker columns if categorical=True ---
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    ("cat",
                     OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                     self.feature_titles)
                ],
                remainder="drop",
                verbose_feature_names_out=False
            )
        else:
            pre = "passthrough"

        # --- Decision Tree classifier ---
        dt = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

        # --- Build pipeline: [preprocessor] -> DT ---
        self.pipeline = _P([("pre", pre), ("clf", dt)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)

        # Build normalized-name map for robust identify()
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       feature_titles: List[str] | None = None,
                       test_size: float = 0.2,
                       drop_singletons: bool = True) -> "DT":
        """
        Train a Decision Tree in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (e.g. from pd.get_dummies)
          - feature_titles are 'marker=value' names, e.g.:
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', 'yp1108ms45=nan', ...
          - No additional OneHotEncoder is applied; features are used as-is.

        This mode is for backward compatibility with older 'combined' models.
        """

        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
            from sklearn.pipeline import Pipeline as _P
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # Auto-generate generic names if not provided
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In 'combined' mode we EXPECT many 'marker=value' titles,
        # so we do NOT raise an error if '=' is present.
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of encoded features (typically 0/1) ---
        # Use float or int; no additional normalization for "nd"/"" here,
        # as X is assumed to be numeric one-hot already.
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)


        # --- Collect unique values per marker from one-hot representation ---
        feature_values: dict[str, list[str]] = {}

        for col_idx, full_name in enumerate(self.feature_titles):
            # Skip values that never occur in the training data (column all zeros)
            if not np.any(X_df.iloc[:, col_idx].values):
                continue

            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                # fallback: treat full_name as marker, with implicit value "1"
                marker, value = full_name, "1"

            feature_values.setdefault(marker, []).append(str(value))

        # Deduplicate + sort
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # --- No extra preprocessing: data is already encoded ---
        pre = "passthrough"

        # --- Decision Tree classifier ---
        dt = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

        # --- Build pipeline: [passthrough] -> DT ---
        self.pipeline = _P([("pre", pre), ("clf", dt)])

        # --- Fit & evaluate ---
        self.pipeline.fit(X_train_df, y_train)
        y_pred_idx = self.pipeline.predict(X_test_df)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = _cm(y_test, y_pred_idx)

        # Sync classes_ with estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)

        # Build normalized-name map for robust identify()
        self._build_norm_feature_map()

        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        return self

########################################################################
class DeltaNonlinLin(MLearn):
    """
    Δ(nonlin−lin) classifier: trains both a non-linear and a linear model
    on the same encoded feature space, and reports the difference in their
    performance.

    - Non-linear model: RandomForestClassifier
    - Linear model:     LogisticRegression

    Identification uses the non-linear model (RandomForest) via the standard
    MLearn.identify/identify_combined machinery.
    """

    def __init__(
        self,
        marker_style: str = "plain",    # 'plain' markers or 'combined' one-hot
        # Non-linear (RF) hyperparameters
        n_estimators: int = 300,
        max_depth: Union[int, None] = None,
        max_features: Union[str, int, float, None] = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        rf_class_weight: Union[str, dict, None] = None,
        # Linear (LR) hyperparameters
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        lr_class_weight: Union[str, dict, None] = None,
        max_iter: int = 1000,
        # Shared
        random_state: int = 42,
        categorical: bool = True,
        min_coverage_warn: float = 0.2,
        raise_on_low_coverage: bool = False,
        n_jobs: int = -1,
    ) -> None:
        try:
            # ensure sklearn is present
            from sklearn.model_selection import train_test_split  # noqa
            from sklearn.pipeline import Pipeline                 # noqa
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        super().__init__(
            model="DNL",
            marker_style=marker_style,
            categorical=categorical,
            min_coverage_warn=min_coverage_warn,
            raise_on_low_coverage=raise_on_low_coverage,
        )

        # RF params
        self.n_estimators    = int(n_estimators)
        self.max_depth       = max_depth
        self.max_features    = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf  = int(min_samples_leaf)
        self.bootstrap       = bool(bootstrap)
        self.rf_class_weight = rf_class_weight
        self.random_state    = random_state
        self.n_jobs          = n_jobs

        # LR params
        self.C              = float(C)
        self.penalty        = penalty
        self.solver         = solver
        self.lr_class_weight = lr_class_weight
        self.lr_max_iter    = int(max_iter)

        # store linear baseline pipeline separately (optional)
        self.baseline_pipeline_: Any | None = None

    # ------------------------------------------------------------------
    # Training dispatcher
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True,
    ) -> "DeltaNonlinLin":
        """
        Dispatching train method for Δ(nonlin−lin).

        If self.marker_style == "combined":
            - X is expected to be ONE-HOT encoded with 'marker=value' titles
              (e.g. 'yp1108ms45=128', 'yp1108ms45=145', ...).
            - Delegates to self.train_combined(...).

        Otherwise (self.marker_style == "plain", default):
            - X must have ONE column per original marker (e.g. '101361', '366141', ...)
            - feature_titles must be those marker names, NOT 'marker=value' names.
            - Category expansion ('0','1','nan','nd','') is handled internally by
              OneHotEncoder in the pipeline.
        """
        style = getattr(self, "marker_style", "plain")
        if style == "combined":
            return self.train_combined(
                X=X,
                y=y,
                feature_titles=feature_titles,
                test_size=test_size,
                drop_singletons=drop_singletons,
            )
        return self._train_plain(
            X=X,
            y=y,
            feature_titles=feature_titles,
            test_size=test_size,
            drop_singletons=drop_singletons,
        )

    # =========================
    # PLAIN MARKER STYLE TRAIN
    # =========================
    def _train_plain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None,
        test_size: float,
        drop_singletons: bool,
    ) -> "DeltaNonlinLin":
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # Auto-generate generic names if not provided
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # --- HARD GUARD against 'marker=value' style in plain mode ---
        bad_titles = [ft for ft in feature_titles if "=" in ft]
        if bad_titles:
            example = ", ".join(bad_titles[:5])
            raise ValueError(
                "DeltaNonlinLin.train(): feature_titles appear to use 'marker=value' style, e.g. "
                f"{example}. This representation is not supported in 'plain' mode.\n"
                "Please pass ONE column per original marker instead "
                "(e.g. '101361', '366141', ...), and let OneHotEncoder "
                "handle category expansion internally."
            )

        # Store clean marker titles, which MUST match what identify() will use
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of normalized symbols ---
        X_df = self._prepare_X_df(X_use, self.feature_titles)


        # --- Collect unique values per marker (no duplicates) ---
        # X_df is already normalized ("" / nd / nan -> ""), so we take it as ground truth.
        self.feature_values = {
            col: sorted(X_df[col].astype(str).unique().tolist())
            for col in self.feature_titles
        }

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles).astype(str)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles).astype(str)

        # --- Preprocessor (shared by both models) ---
        if self.categorical:
            pre = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
                        self.feature_titles,
                    )
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        else:
            pre = "passthrough"

        # --- Non-linear model: RandomForest ---
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            class_weight=self.rf_class_weight,
            oob_score=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # --- Linear model: LogisticRegression ---
        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.lr_max_iter,
            class_weight=self.lr_class_weight,
            multi_class="auto",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        # Pipelines
        rf_pipe = _P([("pre", pre), ("clf", rf)])
        lr_pipe = _P([("pre", pre), ("clf", lr)])

        # --- Fit both models ---
        rf_pipe.fit(X_train_df, y_train)
        lr_pipe.fit(X_train_df, y_train)

        # --- Evaluate ---
        y_pred_rf = rf_pipe.predict(X_test_df)
        y_pred_lr = lr_pipe.predict(X_test_df)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        delta_acc = acc_rf - acc_lr

        report_rf = classification_report(y_test, y_pred_rf, digits=3, zero_division=0)
        report_lr = classification_report(y_test, y_pred_lr, digits=3, zero_division=0)
        cm_rf = _cm(y_test, y_pred_rf).tolist()
        cm_lr = _cm(y_test, y_pred_lr).tolist()

        # Use RF as the primary classifier for identification
        self.pipeline = rf_pipe
        self.baseline_pipeline_ = lr_pipe

        # Sync classes_ with RF estimator
        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)

        # Build normalized-name map for robust identify()
        self._build_norm_feature_map()

        self.training_metrics = {
            "nonlinear": {
                "accuracy": float(acc_rf),
                "classification_report": report_rf,
                "confusion_matrix": cm_rf,
            },
            "linear": {
                "accuracy": float(acc_lr),
                "classification_report": report_lr,
                "confusion_matrix": cm_lr,
            },
            "delta_accuracy": float(delta_acc),
        }
        return self

    # ==================================================
    # COMBINED (ONE-HOT 'marker=value') MARKER STYLE TRAIN
    # ==================================================
    def train_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_titles: List[str] | None = None,
        test_size: float = 0.2,
        drop_singletons: bool = True,
    ) -> "DeltaNonlinLin":
        """
        Train Δ(nonlin−lin) model in 'combined' (one-hot) marker style.

        EXPECTED INPUT:
          - X is already ONE-HOT encoded (e.g. from pd.get_dummies)
          - feature_titles are 'marker=value' names, e.g.:
                'yp1108ms45=128', 'yp1108ms45=145', 'yp1108ms45=nd', 'yp1108ms45=nan', ...
          - No additional OneHotEncoder is applied; features are used as-is.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as _cm
        from sklearn.pipeline import Pipeline as _P

        # --- Validate feature name length ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # Auto-generate generic names if not provided
        if feature_titles is None:
            feature_titles = [f"f{i}" for i in range(X.shape[1])]

        # In 'combined' mode we EXPECT 'marker=value' titles
        self.feature_titles = list(feature_titles)

        # --- Labels, first-seen order preserved ---
        self.titles, y_idx = _order_preserving_label_map(y)

        # --- Optionally drop singleton classes ---
        X_use, y_use = (X, y_idx)
        if drop_singletons:
            X_use, y_use = _drop_singletons(X_use, y_use)

        # --- DataFrame of encoded features (numeric one-hot) ---
        X_df = pd.DataFrame(X_use, columns=self.feature_titles)


        # --- Collect unique values per marker from one-hot representation ---
        feature_values: dict[str, list[str]] = {}

        for col_idx, full_name in enumerate(self.feature_titles):
            # Skip values that never occur in the training data (column all zeros)
            if not np.any(X_df.iloc[:, col_idx].values):
                continue

            if "=" in full_name:
                marker, value = full_name.split("=", 1)
            else:
                # fallback: treat full_name as marker, with implicit value "1"
                marker, value = full_name, "1"

            feature_values.setdefault(marker, []).append(str(value))

        # Deduplicate + sort
        self.feature_values = {m: sorted(set(vals)) for m, vals in feature_values.items()}

        # --- Safe train/test split ---
        X_train_arr, X_test_arr, y_train, y_test = _safe_split(
            X_df.values, y_use, test_size, self.random_state
        )
        X_train_df = pd.DataFrame(X_train_arr, columns=self.feature_titles)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=self.feature_titles)

        # No extra preprocessing: data is already encoded
        pre = "passthrough"

        # --- Non-linear RF ---
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            class_weight=self.rf_class_weight,
            oob_score=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # --- Linear LR ---
        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.lr_max_iter,
            class_weight=self.lr_class_weight,
            multi_class="auto",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        rf_pipe = _P([("pre", pre), ("clf", rf)])
        lr_pipe = _P([("pre", pre), ("clf", lr)])

        # --- Fit & evaluate both ---
        rf_pipe.fit(X_train_df, y_train)
        lr_pipe.fit(X_train_df, y_train)

        y_pred_rf = rf_pipe.predict(X_test_df)
        y_pred_lr = lr_pipe.predict(X_test_df)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        delta_acc = acc_rf - acc_lr

        report_rf = classification_report(y_test, y_pred_rf, digits=3, zero_division=0)
        report_lr = classification_report(y_test, y_pred_lr, digits=3, zero_division=0)
        cm_rf = _cm(y_test, y_pred_rf).tolist()
        cm_lr = _cm(y_test, y_pred_lr).tolist()

        # Use RF as primary classifier
        self.pipeline = rf_pipe
        self.baseline_pipeline_ = lr_pipe

        clf = self.pipeline.named_steps.get("clf", None)
        _set_classes_from_estimator(self, clf, self.titles)
        self._build_norm_feature_map()

        self.training_metrics = {
            "nonlinear": {
                "accuracy": float(acc_rf),
                "classification_report": report_rf,
                "confusion_matrix": cm_rf,
            },
            "linear": {
                "accuracy": float(acc_lr),
                "classification_report": report_lr,
                "confusion_matrix": cm_lr,
            },
            "delta_accuracy": float(delta_acc),
        }
        return self

###############################################################################
# Simple test for loading an existing model (as in your original __main__)
###############################################################################
if __name__ == "__main__":
    def load_model(model_path: str):
        """Load a trained model (.pkl) using joblib/pickle from model_folder/model_name"""
        if not os.path.isfile(model_path):
            sys.exit(f"🛑 Error: Model file not found: {model_path}")
        try:
            import joblib
            model = joblib.load(model_path)
        except Exception:
            import pickle
            with open(model_path, "rb") as fh:
                model = pickle.load(fh)
        return model

    model_path = os.path.join("..", "applications", "yp_classifier_by_genotype", "db", "yp_genotyping.pkl")
    model = load_model(model_path)
    print(getattr(model, "feature_titles", None))
