"""
ml_lib.py
=========
Lightweight, pure-NumPy implementations of the machine-learning primitives
used in this project. Written from scratch so the analysis pipeline can run
on systems that do not have scikit-learn installed.

Contents
--------
- StandardScaler              z-score scaling
- KMeans                      Lloyd's algorithm with k-means++ init
- silhouette_score            average silhouette over a sample
- LogisticRegression          multinomial, L2-regularised, batch gradient descent
- DecisionTreeClassifier      CART with Gini impurity, depth/leaf control
- RandomForestClassifier      bagged decision trees with feature subsampling
- StratifiedKFold             reproducible stratified k-fold splits
- train_test_split            stratified train/test split
- Metrics                     accuracy, macro_f1, balanced_accuracy, roc_auc_ovr,
                              confusion_matrix
- permutation_importance      model-agnostic feature importance

All classes follow a familiar fit / predict / predict_proba interface.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
class StandardScaler:
    """Centre to zero mean and unit variance."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------
class KMeans:
    """K-means clustering with k-means++ initialisation."""

    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 n_init: int = 10, tol: float = 1e-4,
                 random_state: int | None = None) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf

    # ---- helpers --------------------------------------------------------
    def _kmeans_pp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = X.shape[0]
        idx0 = rng.integers(n_samples)
        centers = [X[idx0]]
        for _ in range(1, self.n_clusters):
            d2 = np.min(
                ((X[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(-1),
                axis=1,
            )
            probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n_samples) / n_samples
            idx = rng.choice(n_samples, p=probs)
            centers.append(X[idx])
        return np.stack(centers)

    def _single_run(self, X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.default_rng(seed)
        centers = self._kmeans_pp(X, rng)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(self.max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            new_centers = np.stack([
                X[new_labels == k].mean(axis=0) if (new_labels == k).any()
                else centers[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centers - centers)
            centers, labels = new_centers, new_labels
            if shift < self.tol:
                break
        inertia = ((X - centers[labels]) ** 2).sum()
        return centers, labels, inertia

    # ---- API ------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        base = 0 if self.random_state is None else int(self.random_state)
        best = None
        for i in range(self.n_init):
            centers, labels, inertia = self._single_run(X, base + i)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d2 = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(-1)
        return d2.argmin(axis=1)


class KPrototypes:
    """K-prototypes clustering for mixed binary + numeric data (Huang, 1998).

    Distance to a centroid is the squared Euclidean distance over numeric
    columns plus gamma times the count of mismatches over categorical
    (binary) columns. Centroids are the per-cluster mean for numeric
    columns and the mode for categorical columns. This is the
    methodologically appropriate clustering method when most features are
    binary, where plain K-means' spherical-Gaussian assumption does not
    hold.
    """

    def __init__(self, n_clusters: int = 2, cat_idx: list[int] | None = None,
                 num_idx: list[int] | None = None, gamma: float | None = None,
                 max_iter: int = 100, n_init: int = 10,
                 random_state: int | None = None) -> None:
        self.n_clusters = n_clusters
        self.cat_idx = cat_idx
        self.num_idx = num_idx
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.cost_: float = np.inf
        self.centroids_num_: np.ndarray | None = None
        self.centroids_cat_: np.ndarray | None = None

    def _dist(self, Xn, Xc, cn, cc):
        """Distance from every row to one centroid (cn numeric, cc categorical)."""
        d = np.zeros(len(Xn) if Xn is not None else len(Xc))
        if Xn is not None and Xn.shape[1] > 0:
            d = d + ((Xn - cn) ** 2).sum(axis=1)
        if Xc is not None and Xc.shape[1] > 0:
            d = d + self.gamma * (Xc != cc).sum(axis=1)
        return d

    def _single_run(self, Xn, Xc, seed):
        rng = np.random.default_rng(seed)
        n = len(Xn) if Xn is not None else len(Xc)
        init = rng.choice(n, size=self.n_clusters, replace=False)
        cnum = Xn[init].copy() if Xn is not None else None
        ccat = Xc[init].copy() if Xc is not None else None
        labels = np.zeros(n, dtype=int)
        for _ in range(self.max_iter):
            D = np.stack([self._dist(Xn, Xc, cnum[k] if cnum is not None else None,
                                     ccat[k] if ccat is not None else None)
                          for k in range(self.n_clusters)], axis=1)
            new_labels = D.argmin(axis=1)
            for k in range(self.n_clusters):
                m = new_labels == k
                if not m.any():
                    continue
                if cnum is not None:
                    cnum[k] = Xn[m].mean(axis=0)
                if ccat is not None:
                    # mode per categorical column
                    for j in range(Xc.shape[1]):
                        vals, counts = np.unique(Xc[m, j], return_counts=True)
                        ccat[k, j] = vals[counts.argmax()]
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
        cost = float(sum(
            self._dist(Xn, Xc, cnum[k] if cnum is not None else None,
                       ccat[k] if ccat is not None else None)[labels == k].sum()
            for k in range(self.n_clusters)))
        return labels, cnum, ccat, cost

    def fit(self, X: np.ndarray) -> "KPrototypes":
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        cat_idx = self.cat_idx if self.cat_idx is not None else []
        num_idx = self.num_idx if self.num_idx is not None else \
            [j for j in range(p) if j not in cat_idx]
        Xn = X[:, num_idx] if num_idx else None
        Xc = X[:, cat_idx] if cat_idx else None
        if self.gamma is None:
            # Huang's heuristic: half the mean numeric standard deviation
            self.gamma = 0.5 * float(np.mean(Xn.std(axis=0))) if Xn is not None \
                and Xn.shape[1] > 0 else 1.0
            if self.gamma <= 0:
                self.gamma = 1.0
        base = 0 if self.random_state is None else int(self.random_state)
        best = None
        for i in range(self.n_init):
            labels, cnum, ccat, cost = self._single_run(Xn, Xc, base + i)
            if best is None or cost < best[3]:
                best = (labels, cnum, ccat, cost)
        self.labels_, self.centroids_num_, self.centroids_cat_, self.cost_ = best
        return self


def silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int | None = 2000,
                     random_state: int = 0) -> float:
    """Mean silhouette over (optionally) a random sub-sample for speed."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        X_s, lab_s = X[idx], labels[idx]
    else:
        X_s, lab_s = X, labels
    unique = np.unique(lab_s)
    if len(unique) < 2:
        return 0.0
    s_vals = np.zeros(len(X_s))
    # Pre-compute pairwise distances within the sample (sample_size <= 2000, OK)
    diff = X_s[:, None, :] - X_s[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    for i in range(len(X_s)):
        own = lab_s[i]
        same = lab_s == own
        same[i] = False
        if not same.any():
            s_vals[i] = 0.0
            continue
        a = D[i, same].mean()
        bs = []
        for k in unique:
            if k == own:
                continue
            mask = lab_s == k
            if mask.any():
                bs.append(D[i, mask].mean())
        b = min(bs)
        s_vals[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s_vals.mean())


# ---------------------------------------------------------------------------
# Logistic Regression (multinomial)
# ---------------------------------------------------------------------------
class LogisticRegression:
    """L2-regularised multinomial logistic regression via batch gradient descent."""

    def __init__(self, C: float = 1.0, lr: float = 0.1, max_iter: int = 500,
                 tol: float = 1e-5, random_state: int | None = 0) -> None:
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.W: np.ndarray | None = None  # (n_features+1, n_classes)
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, p = X.shape
        K = len(self.classes_)
        Xb = np.hstack([np.ones((n, 1)), X])
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0, 0.01, size=(p + 1, K))
        Y = np.eye(K)[y_idx]
        lam = 1.0 / max(self.C, 1e-8)
        prev_loss = np.inf
        for it in range(self.max_iter):
            P = self._softmax(Xb @ self.W)
            # log-loss + L2 (don't regularise intercept)
            loss = -np.mean((Y * np.log(P + 1e-12)).sum(axis=1))
            loss += 0.5 * lam * (self.W[1:] ** 2).sum() / n
            grad = Xb.T @ (P - Y) / n
            grad[1:] += lam * self.W[1:] / n
            self.W -= self.lr * grad
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
        return self._softmax(Xb @ self.W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Decision tree (CART, Gini)
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "prob", "n")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.prob: np.ndarray | None = None
        self.n: int = 0


class DecisionTreeClassifier:
    """CART decision tree with Gini impurity."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 20,
                 min_samples_leaf: int = 10, max_features: int | str | None = None,
                 random_state: int | None = None) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None

    def _gini(self, y_idx: np.ndarray, K: int) -> float:
        if len(y_idx) == 0:
            return 0.0
        counts = np.bincount(y_idx, minlength=K) / len(y_idx)
        return 1.0 - (counts ** 2).sum()

    def _best_split(self, X: np.ndarray, y_idx: np.ndarray, K: int,
                    feat_idx: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float] | None:
        n, _ = X.shape
        base_gini = self._gini(y_idx, K)
        best_gain = 0.0
        best_split = None
        for f in feat_idx:
            col = X[:, f]
            # Use up to 32 quantile candidates to bound complexity
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            if len(uniq) > 16:
                qs = np.quantile(col, np.linspace(0.1, 0.9, 16))
                thresholds = np.unique(qs)
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2
            for t in thresholds:
                left_mask = col <= t
                nL = left_mask.sum()
                nR = n - nL
                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue
                gL = self._gini(y_idx[left_mask], K)
                gR = self._gini(y_idx[~left_mask], K)
                gain = base_gini - (nL * gL + nR * gR) / n
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, float(t), gain)
        return best_split

    def _build(self, X: np.ndarray, y_idx: np.ndarray, K: int, depth: int,
               rng: np.random.Generator) -> _Node:
        node = _Node()
        node.n = len(y_idx)
        counts = np.bincount(y_idx, minlength=K)
        node.prob = counts / max(counts.sum(), 1)
        if (depth >= self.max_depth
                or len(y_idx) < self.min_samples_split
                or (counts > 0).sum() == 1):
            return node
        n_feat = X.shape[1]
        if self.max_features is None:
            feat_idx = np.arange(n_feat)
        elif isinstance(self.max_features, int):
            feat_idx = rng.choice(n_feat, size=min(self.max_features, n_feat), replace=False)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_feat)))
            feat_idx = rng.choice(n_feat, size=k, replace=False)
        else:
            feat_idx = np.arange(n_feat)
        split = self._best_split(X, y_idx, K, feat_idx, rng)
        if split is None:
            return node
        f, t, _ = split
        left_mask = X[:, f] <= t
        node.feature = f
        node.threshold = t
        node.left = self._build(X[left_mask], y_idx[left_mask], K, depth + 1, rng)
        node.right = self._build(X[~left_mask], y_idx[~left_mask], K, depth + 1, rng)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y_idx, len(self.classes_), 0, rng)
        return self

    def _walk(self, node: _Node, x: np.ndarray) -> np.ndarray:
        while node.feature is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prob

    def _walk_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised prediction: route an index set through the tree."""
        K = len(self.classes_)
        out = np.zeros((len(X), K))
        # Stack: list of (node, indices)
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.feature is None or len(idx) == 0:
                if len(idx):
                    out[idx] = node.prob
                continue
            mask = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._walk_batch(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class _RegNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "leaf_id")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _RegNode | None = None
        self.right: _RegNode | None = None
        self.value: float = 0.0
        self.leaf_id: int = -1


class DecisionTreeRegressor:
    """CART regression tree with variance-reduction splits.

    Exposes apply() returning the leaf index for each sample, which the
    gradient-boosting classifier uses for Friedman leaf-value refinement.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 20,
                 min_samples_leaf: int = 10, max_features: str | int | None = None,
                 random_state: int | None = None) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _RegNode | None = None
        self._leaf_counter = 0

    @staticmethod
    def _var(y: np.ndarray) -> float:
        return float(y.var()) if len(y) else 0.0

    def _best_split(self, X: np.ndarray, y: np.ndarray,
                    feat_idx: np.ndarray) -> tuple[int, float] | None:
        n = len(y)
        parent_var = self._var(y)
        best_gain = 0.0
        best = None
        for f in feat_idx:
            col = X[:, f]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            if len(uniq) > 16:
                thresholds = np.unique(np.quantile(col, np.linspace(0.1, 0.9, 16)))
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2
            for t in thresholds:
                left = col <= t
                nL = int(left.sum())
                nR = n - nL
                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue
                gain = parent_var - (nL * self._var(y[left])
                                     + nR * self._var(y[~left])) / n
                if gain > best_gain:
                    best_gain = gain
                    best = (f, float(t))
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int,
               rng: np.random.Generator) -> _RegNode:
        node = _RegNode()
        node.value = float(y.mean()) if len(y) else 0.0
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or np.allclose(y, y[0])):
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node
        nf = X.shape[1]
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(nf)))
            feat_idx = rng.choice(nf, size=k, replace=False)
        else:
            feat_idx = np.arange(nf)
        split = self._best_split(X, y, feat_idx)
        if split is None:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node
        f, t = split
        left = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left = self._build(X[left], y[left], depth + 1, rng)
        node.right = self._build(X[~left], y[~left], depth + 1, rng)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._leaf_counter = 0
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y, 0, rng)
        return self

    def _walk(self, x: np.ndarray, want_leaf: bool):
        node = self.root_
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.leaf_id if want_leaf else node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(x, False) for x in X])

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(x, True) for x in X], dtype=int)


class GradientBoostingClassifier:
    """Binary gradient-boosted trees (log-loss) with Friedman leaf refinement.

    Stage-wise additive model: each round fits a regression tree to the
    negative log-loss gradient (y - p); leaf values are then refined by a
    single Newton step gamma = Σresidual / Σp(1-p), the standard TreeBoost
    update. Suited to the binary risk-profile target in this project.
    """

    def __init__(self, n_estimators: int = 80, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_leaf: int = 20,
                 min_samples_split: int = 40, subsample: float = 1.0,
                 random_state: int | None = 0) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.trees_: list[DecisionTreeRegressor] = []
        self.leaf_gammas_: list[dict] = []
        self.classes_: np.ndarray | None = None
        self.F0: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier supports binary targets only")
        y01 = (y == self.classes_[1]).astype(float)
        p0 = float(np.clip(y01.mean(), 1e-6, 1 - 1e-6))
        self.F0 = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.F0)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.trees_, self.leaf_gammas_ = [], []
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            residual = y01 - p  # negative gradient of log-loss
            if self.subsample < 1.0:
                idx = rng.choice(n, size=max(1, int(self.subsample * n)),
                                 replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[idx], residual[idx])
            # Friedman leaf refinement (Newton step) on the fitted subsample
            leaf_ids = tree.apply(X[idx])
            p_sub = p[idx]
            res_sub = residual[idx]
            gammas: dict[int, float] = {}
            for lid in np.unique(leaf_ids):
                m = leaf_ids == lid
                num = res_sub[m].sum()
                den = (p_sub[m] * (1.0 - p_sub[m])).sum()
                gammas[lid] = float(num / (den + 1e-12))
            self.trees_.append(tree)
            self.leaf_gammas_.append(gammas)
            leaves_all = tree.apply(X)
            update = np.array([gammas.get(int(l), 0.0) for l in leaves_all])
            F += self.learning_rate * update
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.F0)
        for tree, gammas in zip(self.trees_, self.leaf_gammas_):
            leaves = tree.apply(X)
            F += self.learning_rate * np.array(
                [gammas.get(int(l), 0.0) for l in leaves])
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._sigmoid(self.decision_function(X))
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class RandomForestClassifier:
    """Random forest with bootstrap aggregation and feature subsampling."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 20, min_samples_leaf: int = 10,
                 max_features: str | int | None = "sqrt",
                 random_state: int | None = 0) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.trees_ = []
        for i in range(self.n_estimators):
            sample_idx = rng.integers(0, n, size=n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Align trees with full class list (some bootstrap samples may miss classes)
        K = len(self.classes_)
        probs = np.zeros((len(X), K))
        for tree in self.trees_:
            p = tree.predict_proba(X)
            # map tree.classes_ into the full class list
            for j, c in enumerate(tree.classes_):
                k = int(np.where(self.classes_ == c)[0][0])
                probs[:, k] += p[:, j]
        probs /= len(self.trees_)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
class StratifiedKFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True,
                 random_state: int | None = 0) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        fold = np.zeros(n, dtype=int)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if self.shuffle:
                rng.shuffle(idx)
            for i, x in enumerate(idx):
                fold[x] = i % self.n_splits
        for k in range(self.n_splits):
            test = np.where(fold == k)[0]
            train = np.where(fold != k)[0]
            yield train, test


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     random_state: int = 0, stratify: np.ndarray | None = None):
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    if stratify is None:
        idx = rng.permutation(len(X))
        cut = int(len(X) * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
    else:
        s = np.asarray(stratify)
        tr_list, te_list = [], []
        for c in np.unique(s):
            ci = np.where(s == c)[0]
            rng.shuffle(ci)
            cut = int(len(ci) * (1 - test_size))
            tr_list.append(ci[:cut])
            te_list.append(ci[cut:])
        tr = np.concatenate(tr_list)
        te = np.concatenate(te_list)
        rng.shuffle(tr)
        rng.shuffle(te)
    return X[tr], X[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    K = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = (2 * tp + fp + fn)
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = []
    for c in np.unique(y_true):
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(((y_pred == c) & mask).sum() / mask.sum())
    return float(np.mean(recalls)) if recalls else 0.0


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal AUC for binary classification."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        return float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr))


def roc_auc_ovr(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    """One-vs-rest macro-averaged AUC."""
    y_true = np.asarray(y_true)
    aucs = []
    for j, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        a = _roc_auc_binary(yb, y_proba[:, j])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------
def permutation_importance(model, X: np.ndarray, y: np.ndarray, n_repeats: int = 5,
                           random_state: int = 0, scoring=macro_f1) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    base = scoring(y, model.predict(X))
    importances = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            importances[j, r] = base - scoring(y, model.predict(X_perm))
    return importances
