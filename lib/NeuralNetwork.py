import sys, os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any

########################################################################
class MLP:  # Multilayer Perceptron (MLP)
    """
    Encapsulates a neural network training pipeline.
    """
    def __init__(self,
                 hidden_layer_sizes=(128, 64),
                 learning_rate_init=1e-3,
                 alpha=1e-4,
                 max_iter=300,
                 batch_size=64,
                 early_stopping=True,
                 n_iter_no_change=10,
                 random_state=42):
        # Import necessary modules
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)
                    
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.titles = []

        self.pipeline: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.training_metrics: dict | None = None
        self.feature_titles: list[str] | None = None   # <--- NEW

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_titles: list[str] | None = None,
              test_size: float = 0.2) -> "NN":
        """
        Train an MLP neural network on feature matrix X and label vector y.
    
        y is expected to be a list/array of string titles (class names).
        This function:
          - builds self.titles = ordered list of unique titles
          - replaces y with integer indices into self.titles for training
          - stores self.classes_ as the string titles for downstream use
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
        # --- 0) Validate feature titles length if provided ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )
    
        # --- 1) Map string titles -> integer indices ---
        y = np.asarray(y).astype(str)
        # preserve first-seen order of titles
        seen = {}
        titles_list = []
        for lbl in y:
            if lbl not in seen:
                seen[lbl] = len(titles_list)
                titles_list.append(lbl)
        self.titles = titles_list                               # store string titles
        y_idx = np.array([seen[lbl] for lbl in y], dtype=int)   # indices for training
    
        # --- 2) Train/test split (stratified if possible) ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, stratify=y_idx, random_state=self.random_state
            )
        except ValueError:
            # fallback without stratification (e.g., class with only 1 sample)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, random_state=self.random_state
            )
    
        # --- 3) Resolve effective batch size ---
        n_train_total = int(X_train.shape[0])
        early = bool(getattr(self, "early_stopping", False))
        val_frac = 0.1 if early else 0.0
        n_fit = (max(1, int(np.floor((1.0 - val_frac) * n_train_total)))
                 if early else n_train_total)
        requested_bs = getattr(self, "batch_size", "auto")
        if requested_bs in (None, "auto"):
            effective_bs = min(200, n_fit)
        else:
            try:
                bs = int(requested_bs)
            except (TypeError, ValueError):
                bs = 200
            effective_bs = max(1, min(bs, n_fit))
        self.effective_batch_size_ = effective_bs
        self.n_fit_samples_ = n_fit
    
        # --- 4) Pipeline ---
        self.pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                learning_rate_init=self.learning_rate_init,
                alpha=self.alpha,
                batch_size=effective_bs,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=early,
                validation_fraction=val_frac,   # used only if early=True
                n_iter_no_change=self.n_iter_no_change,
                verbose=False
            ))
        ])
    
        # --- 5) Fit & evaluate ---
        self.pipeline.fit(X_train, y_train)
        y_pred_idx = self.pipeline.predict(X_test)
    
        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_idx)
    
        # Keep classes_ as string titles for downstream presentation
        self.classes_ = np.array(self.titles, dtype=object)
        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        self.feature_titles = feature_titles
        return self

    def get_feature_titles(self) -> list[str] | None:
        """
        Returns the list of feature titles (column names) if available.
        """
        return self.feature_titles

    def identify(self, markers: Any) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Given sparse markers (feature -> {-2,-1,1,2}), return class probabilities.
        Unspecified features default to 0.
        """
        log = []
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        if not self.feature_titles:
            raise RuntimeError(
                "Feature titles are not available; ensure NN.train(...) was called with feature_titles."
            )
        if not self.titles:
            raise RuntimeError(
                "Titles mapping is missing; ensure NN.train(...) was called with class labels."
            )
    
        # Normalize markers
        mk = self._normalize_markers(markers)
    
        # Build a single-row DataFrame with all feature columns
        row = {ft: 0.0 for ft in self.feature_titles}  # defaults
        for f, val in mk.items():
            if f in row:
                row[f] = float(val)
        X_one = pd.DataFrame([row], columns=self.feature_titles)
    
        # Predict probabilities
        proba = self.predict_proba(X_one.values)[0]  # shape (n_classes,)
        
        # Map integer indices to string titles
        predictions = {
            str(self.titles[i]): float(proba[i]) for i in range(len(self.titles))
        }
    
        return self.format_output(predictions)
        
    def format_output(self, predictions: Dict[Any, float]) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        def get_odd_value(val):
            return val/(1 - val) if val != 1 else 10
        def get_entropy(values: list[float]) -> float:
            """
            Compute entropy of a list of values.
            If the list contains at least one zero, return its length instead.
            """
            n = len(values)
            if n < 2:
                return 0.0
            
            # Rule: if at least one zero is present → return length
            if any(float(v) in (0, 1.0) for v in values):
                return 0.0
        
            # Convert to probabilities
            arr = np.array(values, dtype=float)
            probs = arr / arr.sum()
        
            # Shannon entropy (base 2)
            entropy = -np.sum(probs * np.log2(probs))/np.log2(n)
            return float(entropy)
                   
        predictions = sorted([list(item) for item in predictions.items()], key=lambda ls: ls[1], reverse=True)
        values = [item[1] for item in predictions]
        best_odd = get_odd_value(values[0])
        odds = [1.0] + [best_odd - get_odd_value(values[i]) for i in range(1, len(values))]
        return {"predictions":predictions, "odd-ratios":odds, "entropy":get_entropy(values)}

    def _normalize_markers(self, markers) -> dict:
        """
        Normalize user-supplied markers into a flat {feature: float_value} dict.

        Accepts:
          • dict-like: {"feat": val, ...}
          • iterable of pairs: [(feat, val), ...] or {feat: val}.items()
          • pandas Series/DataFrame:
              - Series: index = feature names; values = marker values
              - DataFrame: expects a single row; columns = feature names
          • Any values convertible to float. None/NaN -> 0.0
          • Clips values to [-2, 2] (the expected coding); strings are coerced.

        Ignores features not present in self.feature_titles (handled later in identify()).
        Returns a plain dict[str, float].
        """
        import numpy as np
        import pandas as pd

        def _to_float(x):
            if x is None:
                return 0.0
            try:
                # common string encodings like "1", "-1", "0", etc.
                v = float(x)
            except Exception:
                # last-resort: treat anything non-numeric as 0
                return 0.0
            # clip to expected coding range
            if np.isnan(v) or np.isinf(v):
                return 0.0
            # your pipeline expects {-2,-1,0,1,2}
            return float(np.clip(v, -2.0, 2.0))

        norm: dict = {}

        # dict-like
        if hasattr(markers, "items"):
            for k, v in markers.items():
                norm[str(k)] = _to_float(v)
            return norm

        # pandas Series
        try:
            import pandas as pd  # noqa
            if isinstance(markers, pd.Series):
                for k, v in markers.items():
                    norm[str(k)] = _to_float(v)
                return norm
            # pandas DataFrame: use a single row; if multiple, use the first row
            if isinstance(markers, pd.DataFrame):
                if markers.shape[0] == 0:
                    return {}
                row = markers.iloc[0]
                for k, v in row.items():
                    norm[str(k)] = _to_float(v)
                return norm
        except Exception:
            pass  # pandas might not be installed; fall through

        # iterable of pairs
        try:
            for item in markers:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    k, v = item
                    norm[str(k)] = _to_float(v)
                else:
                    # if it's just a list of feature names, set them to 1.0
                    norm[str(item)] = 1.0
            return norm
        except TypeError:
            # not iterable → single feature name; mark present = 1.0
            return {str(markers): 1.0}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper; delegates to the classifier inside the pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        clf = self.pipeline.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "predict_proba"):
            raise RuntimeError("Classifier in the pipeline does not support predict_proba().")
        return clf.predict_proba(X)

########################################################################
class RF:
    """
    Encapsulates a Random Forest classification pipeline with the same
    interface as the provided MLP class (train / identify) and the same output format.
    """
    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: Union[int, None] = None,
                 max_features: Union[str, int, float, None] = "auto",
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 bootstrap: bool = True,
                 class_weight: Union[str, dict, None] = None,  # e.g., "balanced"
                 oob_score: bool = False,
                 random_state: int = 42,
                 n_jobs: int = -1):
        # Loading necessary modules
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)
            
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

        self.titles: List[str] = []
        self.pipeline: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.training_metrics: dict | None = None
        self.feature_titles: list[str] | None = None

    # -------------------- PUBLIC API --------------------
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_titles: list[str] | None = None,
              test_size: float = 0.2) -> "RF":
        """
        Train a Random Forest classifier on feature matrix X and label vector y.

        y is expected to be a list/array of string titles (class names).
        This function:
          - builds self.titles = ordered list of unique titles
          - replaces y with integer indices into self.titles for training
          - stores self.classes_ as the string titles for downstream use
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # 0) Validate feature titles length if provided
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # 1) Map string titles -> integer indices (preserve first-seen order)
        y = np.asarray(y).astype(str)
        seen = {}
        titles_list = []
        for lbl in y:
            if lbl not in seen:
                seen[lbl] = len(titles_list)
                titles_list.append(lbl)
        self.titles = titles_list
        y_idx = np.array([seen[lbl] for lbl in y], dtype=int)

        # 2) Train/test split (stratified if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, stratify=y_idx, random_state=self.random_state
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, random_state=self.random_state
            )

        # 3) Pipeline (no scaler—trees are scale-invariant)
        self.pipeline = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                bootstrap=self.bootstrap,
                class_weight=self.class_weight,
                oob_score=self.oob_score,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ))
        ])

        # 4) Fit & evaluate
        self.pipeline.fit(X_train, y_train)
        y_pred_idx = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_idx)

        # Keep classes_ as string titles for downstream presentation
        self.classes_ = np.array(self.titles, dtype=object)
        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        self.feature_titles = feature_titles
        return self

    def get_feature_titles(self) -> list[str] | None:
        return self.feature_titles

    def identify(self, markers: Any) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Given sparse markers (feature -> {-2,-1,1,2}), return class probabilities.
        Unspecified features default to 0.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        if not self.feature_titles:
            raise RuntimeError(
                "Feature titles are not available; ensure RF.train(...) was called with feature_titles."
            )
        if not self.titles:
            raise RuntimeError(
                "Titles mapping is missing; ensure RF.train(...) was called with class labels."
            )

        # Normalize markers
        mk = self._normalize_markers(markers)

        # Build a single-row DataFrame with all feature columns
        row = {ft: 0.0 for ft in self.feature_titles}  # defaults
        for f, val in mk.items():
            if f in row:
                row[f] = float(val)
        X_one = pd.DataFrame([row], columns=self.feature_titles)

        # Predict probabilities
        proba = self.predict_proba(X_one.values)[0]  # shape (n_classes,)

        # Map integer indices to string titles
        predictions = {
            str(self.titles[i]): float(proba[i]) for i in range(len(self.titles))
        }
        return self.format_output(predictions)

    # -------------------- HELPERS (kept identical to MLP API/format) --------------------
    def format_output(self, predictions: Dict[Any, float]) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        def get_odd_value(val):
            return val/(1 - val) if val != 1 else 10
        def get_entropy(values: list[float]) -> float:
            n = len(values)
            if n < 2:
                return 0.0
            if any(float(v) in (0, 1.0) for v in values):
                return 0.0
            arr = np.array(values, dtype=float)
            probs = arr / arr.sum()
            entropy = -np.sum(probs * np.log2(probs)) / np.log2(n)
            return float(entropy)

        predictions = sorted([list(item) for item in predictions.items()],
                             key=lambda ls: ls[1], reverse=True)
        values = [item[1] for item in predictions]
        best_odd = get_odd_value(values[0])
        odds = [1.0] + [best_odd - get_odd_value(values[i]) for i in range(1, len(values))]
        return {"predictions": predictions, "odd-ratios": odds, "entropy": get_entropy(values)}

    def _normalize_markers(self, markers) -> dict:
        """
        Same normalization logic as in the MLP class.
        Converts various inputs into {feature: float_value} clipped to [-2, 2].
        """
        import numpy as np
        import pandas as pd

        def _to_float(x):
            if x is None:
                return 0.0
            try:
                v = float(x)
            except Exception:
                return 0.0
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return float(np.clip(v, -2.0, 2.0))

        norm: dict = {}

        # dict-like
        if hasattr(markers, "items"):
            for k, v in markers.items():
                norm[str(k)] = _to_float(v)
            return norm

        # pandas Series/DataFrame
        try:
            if isinstance(markers, pd.Series):
                for k, v in markers.items():
                    norm[str(k)] = _to_float(v)
                return norm
            if isinstance(markers, pd.DataFrame):
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Delegates to the classifier inside the pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        clf = self.pipeline.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "predict_proba"):
            raise RuntimeError("Classifier in the pipeline does not support predict_proba().")
        return clf.predict_proba(X)

########################################################################
class SCV:
    """
    Encapsulates a Support Vector Machine (SVC) classification pipeline.
    Interface is identical to MLP and RF classes (train / identify).
    """
    def __init__(self,
                 kernel: str = "rbf",
                 C: float = 1.0,
                 gamma: Union[str, float] = "scale",
                 degree: int = 3,
                 probability: bool = True,
                 shrinking: bool = True,
                 tol: float = 1e-3,
                 max_iter: int = -1,
                 random_state: int = 42):
        # --- Load only necessary sklearn modules ---
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.svm import SVC
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        # --- Initialize parameters ---
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.shrinking = shrinking
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

        self.titles: List[str] = []
        self.pipeline: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.training_metrics: dict | None = None
        self.feature_titles: list[str] | None = None

    # -------------------------------------------------------------------------
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_titles: list[str] | None = None,
              test_size: float = 0.2) -> "SCV":
        """
        Train an SVM classifier with probability output on feature matrix X and labels y.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # --- 0) Validate feature titles ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # --- 1) Map string labels to integer indices ---
        y = np.asarray(y).astype(str)
        seen = {}
        titles_list = []
        for lbl in y:
            if lbl not in seen:
                seen[lbl] = len(titles_list)
                titles_list.append(lbl)
        self.titles = titles_list
        y_idx = np.array([seen[lbl] for lbl in y], dtype=int)

        # --- 2) Train/test split ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, stratify=y_idx, random_state=self.random_state
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, random_state=self.random_state
            )

        # --- 3) Build pipeline (SVM needs scaling) ---
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                probability=self.probability,
                shrinking=self.shrinking,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state
            ))
        ])

        # --- 4) Fit and evaluate ---
        self.pipeline.fit(X_train, y_train)
        y_pred_idx = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_idx)

        self.classes_ = np.array(self.titles, dtype=object)
        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        self.feature_titles = feature_titles
        return self

    # -------------------------------------------------------------------------
    def identify(self, markers: Any) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Given sparse markers (feature -> {-2,-1,1,2}), return class probabilities.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        if not self.feature_titles:
            raise RuntimeError(
                "Feature titles are not available; ensure SCV.train(...) was called with feature_titles."
            )
        if not self.titles:
            raise RuntimeError(
                "Titles mapping is missing; ensure SCV.train(...) was called with class labels."
            )

        mk = self._normalize_markers(markers)
        row = {ft: 0.0 for ft in self.feature_titles}
        for f, val in mk.items():
            if f in row:
                row[f] = float(val)
        X_one = pd.DataFrame([row], columns=self.feature_titles)

        # Predict probabilities
        proba = self.predict_proba(X_one.values)[0]
        predictions = {
            str(self.titles[i]): float(proba[i]) for i in range(len(self.titles))
        }
        return self.format_output(predictions)

    # -------------------------------------------------------------------------
    def format_output(self, predictions: Dict[Any, float]) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        def get_odd_value(val): return val/(1 - val) if val != 1 else 10
        def get_entropy(values: list[float]) -> float:
            n = len(values)
            if n < 2: return 0.0
            if any(float(v) in (0, 1.0) for v in values): return 0.0
            arr = np.array(values, dtype=float)
            probs = arr / arr.sum()
            return float(-np.sum(probs * np.log2(probs)) / np.log2(n))

        predictions = sorted([list(item) for item in predictions.items()],
                             key=lambda ls: ls[1], reverse=True)
        values = [item[1] for item in predictions]
        best_odd = get_odd_value(values[0])
        odds = [1.0] + [best_odd - get_odd_value(values[i]) for i in range(1, len(values))]
        return {"predictions": predictions, "odd-ratios": odds, "entropy": get_entropy(values)}

    # -------------------------------------------------------------------------
    def _normalize_markers(self, markers) -> dict:
        import numpy as np, pandas as pd
        def _to_float(x):
            if x is None: return 0.0
            try: v = float(x)
            except Exception: return 0.0
            if np.isnan(v) or np.isinf(v): return 0.0
            return float(np.clip(v, -2.0, 2.0))

        norm: dict = {}
        if hasattr(markers, "items"):
            for k, v in markers.items():
                norm[str(k)] = _to_float(v)
            return norm
        try:
            if isinstance(markers, pd.Series):
                for k, v in markers.items(): norm[str(k)] = _to_float(v)
                return norm
            if isinstance(markers, pd.DataFrame):
                if markers.shape[0] == 0: return {}
                row = markers.iloc[0]
                for k, v in row.items(): norm[str(k)] = _to_float(v)
                return norm
        except Exception:
            pass
        try:
            for item in markers:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    k, v = item; norm[str(k)] = _to_float(v)
                else:
                    norm[str(item)] = 1.0
            return norm
        except TypeError:
            return {str(markers): 1.0}

    # -------------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        clf = self.pipeline.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "predict_proba"):
            raise RuntimeError("Classifier in the pipeline does not support predict_proba().")
        return clf.predict_proba(X)
        
########################################################################
class LR:
    """
    Encapsulates a Logistic Regression classification pipeline.
    Interface is identical to MLP, RF, and SCV classes (train / identify).
    """
    def __init__(self,
                 penalty: str = "l2",
                 C: float = 1.0,
                 solver: str = "lbfgs",
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 class_weight: Union[str, dict, None] = None,
                 multi_class: str = "auto",
                 random_state: int = 42):
        # --- Load only necessary sklearn modules ---
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.linear_model import LogisticRegression
        except ModuleNotFoundError as e:
            print(f"❌ Missing required module: {e.name}")
            print("➡️  Please install scikit-learn using: pip install scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        # --- Initialize parameters ---
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.random_state = random_state

        self.titles: List[str] = []
        self.pipeline: Pipeline | None = None
        self.classes_: np.ndarray | None = None
        self.training_metrics: dict | None = None
        self.feature_titles: list[str] | None = None

    # -------------------------------------------------------------------------
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_titles: list[str] | None = None,
              test_size: float = 0.2) -> "LR":
        """
        Train a Logistic Regression classifier on feature matrix X and label vector y.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # --- 0) Validate feature titles ---
        if feature_titles is not None and len(feature_titles) != X.shape[1]:
            raise ValueError(
                f"feature_titles length ({len(feature_titles)}) != X.shape[1] ({X.shape[1]})"
            )

        # --- 1) Map string labels to integer indices ---
        y = np.asarray(y).astype(str)
        seen = {}
        titles_list = []
        for lbl in y:
            if lbl not in seen:
                seen[lbl] = len(titles_list)
                titles_list.append(lbl)
        self.titles = titles_list
        y_idx = np.array([seen[lbl] for lbl in y], dtype=int)

        # --- 2) Train/test split ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, stratify=y_idx, random_state=self.random_state
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_idx, test_size=test_size, random_state=self.random_state
            )

        # --- 3) Build pipeline (standardize features) ---
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
                multi_class=self.multi_class,
                random_state=self.random_state
            ))
        ])

        # --- 4) Fit and evaluate ---
        self.pipeline.fit(X_train, y_train)
        y_pred_idx = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred_idx)
        report = classification_report(y_test, y_pred_idx, digits=3, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_idx)

        self.classes_ = np.array(self.titles, dtype=object)
        self.training_metrics = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        self.feature_titles = feature_titles
        return self

    # -------------------------------------------------------------------------
    def identify(self, markers: Any) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        """
        Given sparse markers (feature -> {-2,-1,1,2}), return class probabilities.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        if not self.feature_titles:
            raise RuntimeError(
                "Feature titles are not available; ensure LR.train(...) was called with feature_titles."
            )
        if not self.titles:
            raise RuntimeError(
                "Titles mapping is missing; ensure LR.train(...) was called with class labels."
            )

        mk = self._normalize_markers(markers)
        row = {ft: 0.0 for ft in self.feature_titles}
        for f, val in mk.items():
            if f in row:
                row[f] = float(val)
        X_one = pd.DataFrame([row], columns=self.feature_titles)

        # Predict probabilities
        proba = self.predict_proba(X_one.values)[0]
        predictions = {
            str(self.titles[i]): float(proba[i]) for i in range(len(self.titles))
        }
        return self.format_output(predictions)

    # -------------------------------------------------------------------------
    def format_output(self, predictions: Dict[Any, float]) -> Dict[str, Union[List[List[Union[str, float]]], List[float], float]]:
        def get_odd_value(val): return val/(1 - val) if val != 1 else 10
        def get_entropy(values: list[float]) -> float:
            n = len(values)
            if n < 2: return 0.0
            if any(float(v) in (0, 1.0) for v in values): return 0.0
            arr = np.array(values, dtype=float)
            probs = arr / arr.sum()
            return float(-np.sum(probs * np.log2(probs)) / np.log2(n))

        predictions = sorted([list(item) for item in predictions.items()],
                             key=lambda ls: ls[1], reverse=True)
        values = [item[1] for item in predictions]
        best_odd = get_odd_value(values[0])
        odds = [1.0] + [best_odd - get_odd_value(values[i]) for i in range(1, len(values))]
        return {"predictions": predictions, "odd-ratios": odds, "entropy": get_entropy(values)}

    # -------------------------------------------------------------------------
    def _normalize_markers(self, markers) -> dict:
        import numpy as np, pandas as pd
        def _to_float(x):
            if x is None: return 0.0
            try: v = float(x)
            except Exception: return 0.0
            if np.isnan(v) or np.isinf(v): return 0.0
            return float(np.clip(v, -2.0, 2.0))

        norm: dict = {}
        if hasattr(markers, "items"):
            for k, v in markers.items():
                norm[str(k)] = _to_float(v)
            return norm
        try:
            if isinstance(markers, pd.Series):
                for k, v in markers.items(): norm[str(k)] = _to_float(v)
                return norm
            if isinstance(markers, pd.DataFrame):
                if markers.shape[0] == 0: return {}
                row = markers.iloc[0]
                for k, v in row.items(): norm[str(k)] = _to_float(v)
                return norm
        except Exception:
            pass
        try:
            for item in markers:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    k, v = item; norm[str(k)] = _to_float(v)
                else:
                    norm[str(item)] = 1.0
            return norm
        except TypeError:
            return {str(markers): 1.0}

    # -------------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        clf = self.pipeline.named_steps.get("clf", None)
        if clf is None or not hasattr(clf, "predict_proba"):
            raise RuntimeError("Classifier in the pipeline does not support predict_proba().")
        return clf.predict_proba(X)

########################################################################
class MBCS:
    """
    Matrix-Based Classifier Search:
      - Analyzes X (features) + y (labels)
      - Computes dataset/clustering descriptors
      - Runs quick CV probes (LR, LinearSVC, SVC-RBF, RF, small MLP)
      - Recommends one of: 'LR', 'RF', 'SVC', 'MLP'
    """

    def __init__(self,
                 cv_splits: int = 5,
                 random_state: int = 42,
                 run_mlp_probe: bool = True):
        # --- Load only sklearn modules used by this class ---
        try:
            # model selection / pipeline / preprocessing
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            # clustering + metrics
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            # classifiers
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC, LinearSVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
        except ModuleNotFoundError as e:
            print(f"Missing required module: {e.name}\nInstall scikit-learn: pip install -U scikit-learn")
            sys.exit(1)
        except ImportError as e:
            print(f"Import error: {e}")
            sys.exit(1)

        # store refs (so we can instantiate later without re-importing)
        self._Pipeline = Pipeline
        self._StandardScaler = StandardScaler
        self._StratifiedKFold = StratifiedKFold
        self._cross_val_score = cross_val_score

        self._KMeans = KMeans
        self._silhouette_score = silhouette_score
        self._calinski_harabasz_score = calinski_harabasz_score
        self._davies_bouldin_score = davies_bouldin_score

        self._LogisticRegression = LogisticRegression
        self._SVC = SVC
        self._LinearSVC = LinearSVC
        self._RandomForestClassifier = RandomForestClassifier
        self._MLPClassifier = MLPClassifier

        # config
        self.cv_splits = int(cv_splits)
        self.random_state = int(random_state)
        self.run_mlp_probe = bool(run_mlp_probe)

        # results
        self.recommendation: str | None = None
        self.rationale: list[str] | None = None
        self.dataset_summary: dict | None = None
        self.clustering_scores: dict | None = None
        self.probe_scores: dict | None = None
        self.n_classes_: int | None = None

    # --------------------------- PUBLIC API ---------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> "MBCS":
        """
        Analyze dataset and compute recommendation. Stores results in fields and returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        desc = self._basic_stats(X, y)
        clus = self._cluster_scores(X, max(2, desc["n_classes"]))
        probes = self._probe_models(X, y)

        # derive signals
        linear_score = np.nanmean([probes.get("LR", np.nan), probes.get("LinearSVC", np.nan)])
        nonlinear_score = probes.get("SVC_RBF", np.nan)
        rf_score = probes.get("RF", np.nan)
        mlp_score = probes.get("MLP_small", np.nan)
        delta_nonlinear = (nonlinear_score - linear_score
                           if (not np.isnan(nonlinear_score) and not np.isnan(linear_score)) else np.nan)

        n, p = desc["n_samples"], desc["n_features"]
        large_data = (n >= 2000) or (p >= 1000)
        very_sparse = desc["sparsity"] > 0.8
        noisy_or_missing = (desc["missing_frac"] > 0.0) or very_sparse
        small_medium = n < 10000

        # decision rules
        rationale: list[str] = []
        if not np.isnan(linear_score) and (linear_score >= 0.85) and (np.isnan(delta_nonlinear) or delta_nonlinear < 0.03):
            rec = "LR"
            rationale.append("High linear probe accuracy; nonlinear kernel brings negligible gain.")
        elif not np.isnan(delta_nonlinear) and (delta_nonlinear >= 0.05) and small_medium:
            if large_data and (not np.isnan(mlp_score)) and (mlp_score >= (nonlinear_score - 0.01)):
                rec = "MLP"
                rationale.append("Nonlinear boundary and large/high-dim data; MLP scales better than SVC.")
            else:
                rec = "SVC"
                rationale.append("Nonlinear boundary indicated on small/medium dataset; SVC (RBF) suitable.")
        elif noisy_or_missing or (not np.isnan(rf_score) and rf_score >=
                                  max([x for x in [linear_score, nonlinear_score, mlp_score] if not np.isnan(x)] + [-np.inf]) - 0.01):
            rec = "RF"
            rationale.append("Data look noisy/sparse or RF competitive; choose robust tree ensemble.")
        else:
            # fallback to best observed probe
            best = {
                "LR": linear_score,
                "SVC": nonlinear_score,
                "RF": rf_score,
                "MLP": mlp_score
            }
            rec = max(best, key=lambda k: (best[k] if not np.isnan(best[k]) else -np.inf))
            rationale.append("Selected the best cross-validated probe among candidates.")

        # persist
        self.recommendation = rec
        self.rationale = rationale
        self.dataset_summary = desc
        self.clustering_scores = clus
        self.probe_scores = {
            "LR": linear_score,
            "SVC_RBF": nonlinear_score,
            "RF": rf_score,
            "MLP_small": mlp_score,
            "delta_nonlinear_minus_linear": delta_nonlinear
        }
        self.n_classes_ = desc["n_classes"]
        return self

    def identify(self) -> dict:
        """
        Return the computed recommendation and diagnostics (no input required).
        Kept for API parity with previous classes (call after train()).
        """
        if self.recommendation is None:
            raise RuntimeError("MBCS not trained yet. Call train(X, y) first.")
        return {
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "dataset_summary": self.dataset_summary,
            "clustering_scores": self.clustering_scores,
            "probe_scores": self.probe_scores
        }

    # --------------------------- HELPERS ---------------------------
    def _basic_stats(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples, n_features = X.shape
        # missingness & sparsity
        if np.issubdtype(X.dtype, np.number):
            has_nan = np.isnan(X).any()
            missing_frac = float(np.isnan(X).mean()) if has_nan else 0.0
            zeros = int((X == 0).sum()) if n_samples * n_features > 0 else 0
            sparsity = zeros / float(n_samples * n_features) if n_samples * n_features > 0 else 0.0
        else:
            missing_frac, sparsity = 0.0, 0.0

        # class counts / imbalance
        uniq, counts = np.unique(y, return_counts=True)
        class_counts = {str(k): int(v) for k, v in zip(uniq, counts)}
        imbalance_ratio = float(counts.max() / max(1, counts.min()))
        return {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_classes": int(len(uniq)),
            "class_counts": class_counts,
            "imbalance_ratio": float(imbalance_ratio),
            "missing_frac": float(missing_frac),
            "sparsity": float(sparsity),
        }

    def _cluster_scores(self, X: np.ndarray, n_clusters: int) -> dict:
        scaler = self._StandardScaler()
        Xs = scaler.fit_transform(X)
        km = self._KMeans(n_clusters=n_clusters, n_init=5, random_state=self.random_state)
        try:
            labels = km.fit_predict(Xs)
        except Exception:
            return {"silhouette": float("nan"),
                    "calinski_harabasz": float("nan"),
                    "davies_bouldin": float("nan")}
        out = {}
        try:
            out["silhouette"] = float(self._silhouette_score(Xs, labels))
        except Exception:
            out["silhouette"] = float("nan")
        try:
            out["calinski_harabasz"] = float(self._calinski_harabasz_score(Xs, labels))
        except Exception:
            out["calinski_harabasz"] = float("nan")
        try:
            out["davies_bouldin"] = float(self._davies_bouldin_score(Xs, labels))
        except Exception:
            out["davies_bouldin"] = float("nan")
        return out

    def _cv_score(self, est, X, y) -> float:
        # ensure at least 2 folds if very few classes
        n_classes = max(1, np.unique(y).size)
        n_splits = min(self.cv_splits, max(2, n_classes))
        cv = self._StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        try:
            scores = self._cross_val_score(est, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            return float(np.mean(scores))
        except Exception:
            return float("nan")

    def _probe_models(self, X: np.ndarray, y: np.ndarray) -> dict:
        probes = {}
        # LR
        lr = self._Pipeline([
            ("scaler", self._StandardScaler()),
            ("clf", self._LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto"))
        ])
        # Linear SVC (margin-based linearity check)
        linsvc = self._Pipeline([
            ("scaler", self._StandardScaler()),
            ("clf", self._LinearSVC(C=1.0, tol=1e-3))
        ])
        # SVC with RBF kernel
        svc_rbf = self._Pipeline([
            ("scaler", self._StandardScaler()),
            ("clf", self._SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))
        ])
        # Random Forest (robust to noise/mixed types)
        rf = self._RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=self.random_state
        )
        # Small MLP (optional)
        if self.run_mlp_probe:
            mlp = self._Pipeline([
                ("scaler", self._StandardScaler()),
                ("clf", self._MLPClassifier(hidden_layer_sizes=(64,),
                                            max_iter=200, alpha=1e-4,
                                            learning_rate_init=1e-3,
                                            random_state=self.random_state))
            ])
        # Run CV
        probes["LR"] = self._cv_score(lr, X, y)
        probes["LinearSVC"] = self._cv_score(linsvc, X, y)
        probes["SVC_RBF"] = self._cv_score(svc_rbf, X, y)
        probes["RF"] = self._cv_score(rf, X, y)
        if self.run_mlp_probe:
            probes["MLP_small"] = self._cv_score(mlp, X, y)
        else:
            probes["MLP_small"] = float("nan")
        return probes
