from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import math
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from .data import model_columns, apply_group_state
from .config import DATASETS


MISSING_TOKEN = "__MISSING__"


@dataclass
class FairConfig:
    lambda_fair: float = 1.0
    lambda_u: float = 0.5
    alpha: float = 1.0
    eta: float = 0.05
    epsilon: float = 0.025
    update_batch: int = 512
    candidate_count: int = 16
    constraint_mode: str = "joint"  # joint, dp, eo
    adapt: bool = True
    group_mode: str = "joint"       # joint, marginal
    penalty_mode: str = "incremental"  # incremental, absolute
    feature_correction: bool = True


class ColumnCodec:
    def __init__(self, series: pd.Series):
        self.name = series.name
        self.numeric = pd.api.types.is_numeric_dtype(series)
        if self.numeric:
            x = pd.to_numeric(series, errors="coerce")
            self.median = float(x.median()) if x.notna().any() else 0.0
            self.categories = None
            self.mapping = None
        else:
            s = series.astype("string").fillna(MISSING_TOKEN)
            cats = sorted(map(str, s.unique()))
            self.categories = cats
            self.mapping = {v: i for i, v in enumerate(cats)}
            self.median = None

    def encode(self, values) -> np.ndarray:
        s = pd.Series(np.asarray(values).reshape(-1))
        if self.numeric:
            return pd.to_numeric(s, errors="coerce").fillna(self.median).to_numpy(dtype=np.float32)
        ss = s.astype("string").fillna(MISSING_TOKEN).astype(str)
        return ss.map(lambda v: self.mapping.get(v, -1)).to_numpy(dtype=np.float32)


class AutoRegressiveTreeGenerator:
    """
    Autoregressive CART generator with fairness-aware soft leaf resampling.

    Base GDT:
      z_t is sampled from the empirical training values in the CART leaf
      reached by z_<t; the target is generated last from a target CART.

    Full MultiFair:
      for the last source column needed to define the requested k-way
      intersection, C donor candidates are drawn from the leaf proposal q.
      Their weights use a prefix outcome surrogate plus online group statistics.
      This is the feature-level leaf correction. The target conditional is then
      reweighted using the realized intersection and DP/EO constraints.

    Target-only MultiFair is retained as an ablation by setting
    FairConfig.feature_correction=False.
    """

    def __init__(self, dataset: str, max_depth: int = 7, min_samples_leaf: int = 40, seed: int = 42):
        self.dataset = dataset
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.feature_cols: List[str] = []
        self.codecs: Dict[str, ColumnCodec] = {}
        self.trees: List[Any] = []
        self.leaf_members: List[Dict[int, np.ndarray]] = []
        self.train_values: Dict[str, np.ndarray] = {}
        self.train_y: np.ndarray | None = None
        self.target_tree = None
        self.target_leaf_prob: Dict[int, float] = {}
        self.prefix_target_trees: Dict[int, Any] = {}
        self.n_train = 0
        self.fit_seconds = 0.0

    @staticmethod
    def _positive_proba(tree, X):
        p = tree.predict_proba(X)
        classes = list(tree.classes_)
        if 1 in classes:
            return p[:, classes.index(1)]
        # Degenerate training subset. Use its sole class deterministically.
        return np.ones(len(X), dtype=float) if classes and int(classes[0]) == 1 else np.zeros(len(X), dtype=float)

    def fit(self, train_df: pd.DataFrame):
        t0 = time.perf_counter()
        self.feature_cols = model_columns(train_df, self.dataset)
        self.n_train = len(train_df)

        for c in self.feature_cols:
            self.codecs[c] = ColumnCodec(train_df[c])
            self.train_values[c] = train_df[c].to_numpy(copy=True)

        p = len(self.feature_cols)
        Z = np.empty((len(train_df), p), dtype=np.float32)
        for j, c in enumerate(self.feature_cols):
            Z[:, j] = self.codecs[c].encode(train_df[c])

        self.trees = []
        self.leaf_members = []

        for j, c in enumerate(self.feature_cols):
            if j == 0:
                self.trees.append(None)
                self.leaf_members.append({0: np.arange(len(train_df), dtype=np.int32)})
                continue

            yenc = Z[:, j]
            codec = self.codecs[c]
            if codec.numeric:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.seed + j,
                )
            else:
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.seed + j,
                    class_weight=None,
                )

            tree.fit(Z[:, :j], yenc)
            leaves = tree.apply(Z[:, :j])
            members = {}
            for leaf in np.unique(leaves):
                members[int(leaf)] = np.flatnonzero(leaves == leaf).astype(np.int32)
            self.trees.append(tree)
            self.leaf_members.append(members)

        y = pd.to_numeric(train_df["__target__"], errors="coerce").astype(int).to_numpy()
        self.train_y = y.astype(np.int8)

        self.target_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed + 999,
        )
        self.target_tree.fit(Z, y)
        target_leaves = self.target_tree.apply(Z)
        self.target_leaf_prob = {}
        self.global_target_p = float(np.mean(y))
        for leaf in np.unique(target_leaves):
            yy = y[target_leaves == leaf]
            self.target_leaf_prob[int(leaf)] = float((yy.sum() + 0.5) / (len(yy) + 1.0))

        # Prefix target surrogates only at group-defining source positions.
        # They replace the much heavier HistGradientBoosting proxy used in v1.
        source_positions = []
        for src in DATASETS[self.dataset]["group_sources"]:
            if src in self.feature_cols:
                source_positions.append(self.feature_cols.index(src))
        self.prefix_target_trees = {}
        for j in sorted(set(source_positions)):
            pt = DecisionTreeClassifier(
                max_depth=min(5, self.max_depth),
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed + 5000 + j,
            )
            pt.fit(Z[:, : j + 1], y)
            self.prefix_target_trees[j] = pt

        self.fit_seconds = time.perf_counter() - t0
        return self

    def _sample_features(self, n: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, np.ndarray]:
        p = len(self.feature_cols)
        Zg = np.empty((n, p), dtype=np.float32)
        raw = {}

        for j, c in enumerate(self.feature_cols):
            if j == 0:
                member_idx = self.leaf_members[j][0]
                picked = rng.choice(member_idx, size=n, replace=True)
                vals = self.train_values[c][picked]
            else:
                tree = self.trees[j]
                leaves = tree.apply(Zg[:, :j])
                vals = np.empty(n, dtype=object)
                for leaf in np.unique(leaves):
                    pos = np.flatnonzero(leaves == leaf)
                    pool = self.leaf_members[j].get(int(leaf))
                    if pool is None or len(pool) == 0:
                        pool = np.arange(self.n_train, dtype=np.int32)
                    picked = rng.choice(pool, size=len(pos), replace=True)
                    vals[pos] = self.train_values[c][picked]

                if self.codecs[c].numeric:
                    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()

            raw[c] = vals
            Zg[:, j] = self.codecs[c].encode(vals)

        return pd.DataFrame(raw), Zg

    def _group_component(self, group_idx: int, values, group_state: Dict[str, Any]) -> np.ndarray:
        spec = DATASETS[self.dataset]
        source = spec["group_sources"][group_idx]
        s = pd.Series(np.asarray(values).reshape(-1))

        if self.dataset == "communities_crime":
            e = np.asarray(group_state["quantile_edges"][source], dtype=float)
            labels = [f"Q{i+1}" for i in range(len(e) - 1)]
            out = pd.cut(
                pd.to_numeric(s, errors="coerce"),
                bins=e,
                labels=labels,
                include_lowest=True,
                right=True,
            ).astype("string")
        elif self.dataset == "german_credit" and group_idx == 0:
            out = s.astype("string").str.strip().map(spec["sex_map"]).astype("string")
        elif source == spec.get("age_source") and "age_bins" in spec:
            out = pd.cut(
                pd.to_numeric(s, errors="coerce"),
                bins=spec["age_bins"],
                labels=spec["age_labels"],
                right=False,
                include_lowest=True,
            ).astype("string")
        else:
            out = s.astype("string").str.strip()
            if self.dataset == "diabetes130" and group_idx == 0:
                out = out.where(out.isin(["Male", "Female"]), pd.NA)

        return out.fillna(MISSING_TOKEN).astype(str).to_numpy()

    def _candidate_group_codes(
        self,
        raw: Dict[str, np.ndarray],
        current_col: str,
        cand_vals: np.ndarray,
        k: int,
        group_state: Dict[str, Any],
        group_mode: str,
    ):
        n, C = cand_vals.shape
        spec = DATASETS[self.dataset]
        components = []
        for gi, src in enumerate(spec["group_sources"][:k]):
            if src == current_col:
                vals = cand_vals.reshape(-1)
            else:
                if src not in raw:
                    raise RuntimeError(
                        f"Feature correction called before group source {src!r} was generated"
                    )
                vals = np.repeat(np.asarray(raw[src]), C)
            comp = self._group_component(gi, vals, group_state).reshape(n, C)
            components.append(comp)

        if group_mode == "joint":
            keys = components[0].astype(object)
            for comp in components[1:]:
                keys = np.char.add(np.char.add(keys.astype(str), "||"), comp.astype(str))
            codes, uniques = pd.factorize(keys.reshape(-1), sort=True)
            return [(codes.reshape(n, C).astype(np.int32), len(uniques))]

        result = []
        for comp in components:
            codes, uniques = pd.factorize(comp.reshape(-1), sort=True)
            result.append((codes.reshape(n, C).astype(np.int32), len(uniques)))
        return result

    def _sample_features_fair(
        self,
        n: int,
        rng: np.random.Generator,
        group_state: Dict[str, Any],
        k: int,
        cfg: FairConfig,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, float]]:
        p = len(self.feature_cols)
        Zg = np.empty((n, p), dtype=np.float32)
        raw: Dict[str, np.ndarray] = {}
        spec = DATASETS[self.dataset]
        selected_sources = spec["group_sources"][:k]
        positions = [self.feature_cols.index(src) for src in selected_sources if src in self.feature_cols]
        if not positions:
            X, Z = self._sample_features(n, rng)
            return X, Z, {"feature_lambda_dp": 0.0, "feature_lambda_eo": 0.0}
        j_star = max(positions)
        C = max(2, int(cfg.candidate_count))

        feature_meta = {"feature_lambda_dp": 0.0, "feature_lambda_eo": 0.0}

        for j, c in enumerate(self.feature_cols):
            if j != j_star:
                if j == 0:
                    pool = self.leaf_members[j][0]
                    picked = rng.choice(pool, size=n, replace=True)
                    vals = self.train_values[c][picked]
                else:
                    leaves = self.trees[j].apply(Zg[:, :j])
                    vals = np.empty(n, dtype=object)
                    for leaf in np.unique(leaves):
                        pos = np.flatnonzero(leaves == leaf)
                        pool = self.leaf_members[j].get(int(leaf))
                        if pool is None or len(pool) == 0:
                            pool = np.arange(self.n_train, dtype=np.int32)
                        picked = rng.choice(pool, size=len(pos), replace=True)
                        vals[pos] = self.train_values[c][picked]
                    if self.codecs[c].numeric:
                        vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
                raw[c] = vals
                Zg[:, j] = self.codecs[c].encode(vals)
                continue

            # Candidate matrix drawn from the empirical leaf proposal q.
            cand_idx = np.empty((n, C), dtype=np.int32)
            if j == 0:
                pool = self.leaf_members[0][0]
                cand_idx[:] = rng.choice(pool, size=(n, C), replace=True)
            else:
                leaves = self.trees[j].apply(Zg[:, :j])
                for leaf in np.unique(leaves):
                    pos = np.flatnonzero(leaves == leaf)
                    pool = self.leaf_members[j].get(int(leaf))
                    if pool is None or len(pool) == 0:
                        pool = np.arange(self.n_train, dtype=np.int32)
                    cand_idx[pos, :] = rng.choice(pool, size=(len(pos), C), replace=True)

            cand_vals = self.train_values[c][cand_idx]
            cand_enc = self.codecs[c].encode(cand_vals.reshape(-1)).reshape(n, C)

            # Prefix outcome surrogate for candidate utility/fairness effects.
            if j == 0:
                prefix = cand_enc.reshape(-1, 1)
            else:
                prefix = np.concatenate(
                    [np.repeat(Zg[:, :j], C, axis=0), cand_enc.reshape(-1, 1)],
                    axis=1,
                )
            ptree = self.prefix_target_trees[j]
            proxy = self._positive_proba(ptree, prefix).reshape(n, C)
            donor_y = self.train_y[cand_idx]

            code_sets = self._candidate_group_codes(
                raw=raw,
                current_col=c,
                cand_vals=cand_vals,
                k=k,
                group_state=group_state,
                group_mode=cfg.group_mode,
            )
            counters = [_FeatureCounter(G, cfg.alpha) for _, G in code_sets]

            lam_dp = float(cfg.lambda_fair) if cfg.constraint_mode in ("joint", "dp") else 0.0
            lam_eo = float(cfg.lambda_fair) if cfg.constraint_mode in ("joint", "eo") else 0.0
            selected = np.empty(n, dtype=np.int32)

            global_state = [0, 0.0, 0, 0.0, 0, 0.0]
            for i in range(n):
                base_pred = float(np.mean(proxy[i]))
                logw = np.empty(C, dtype=float)
                for h in range(C):
                    dp_cost = 0.0
                    eo_cost = 0.0
                    for counter, (codes, _) in zip(counters, code_sets):
                        dpc, eoc = counter.cost(
                            int(codes[i, h]),
                            int(donor_y[i, h]),
                            float(proxy[i, h]),
                            tuple(global_state),
                            mode=cfg.penalty_mode,
                        )
                        dp_cost += dpc
                        eo_cost += eoc
                    dp_cost /= max(1, len(counters))
                    eo_cost /= max(1, len(counters))
                    utility_cost = abs(float(proxy[i, h]) - base_pred)
                    logw[h] = -cfg.lambda_u * utility_cost - lam_dp * dp_cost - lam_eo * eo_cost

                m = float(np.max(logw))
                probs = np.exp(logw - m)
                probs /= probs.sum()
                hsel = int(rng.choice(C, p=probs))
                selected[i] = hsel

                yy = int(donor_y[i, hsel])
                pp = float(proxy[i, hsel])
                for counter, (codes, _) in zip(counters, code_sets):
                    counter.update(int(codes[i, hsel]), yy, pp)

                global_state[0] += 1
                global_state[1] += pp
                if yy == 1:
                    global_state[2] += 1
                    global_state[3] += pp
                else:
                    global_state[4] += 1
                    global_state[5] += pp

                if cfg.adapt and (i + 1) % int(cfg.update_batch) == 0:
                    vals = [counter.violations(tuple(global_state)) for counter in counters]
                    dp_v = float(np.mean([v[0] for v in vals]))
                    eo_v = float(np.mean([v[1] for v in vals]))
                    if cfg.constraint_mode in ("joint", "dp"):
                        lam_dp = max(0.0, lam_dp + cfg.eta * (dp_v - cfg.epsilon))
                    if cfg.constraint_mode in ("joint", "eo"):
                        lam_eo = max(0.0, lam_eo + cfg.eta * (eo_v - cfg.epsilon))

            rows = np.arange(n)
            vals = cand_vals[rows, selected]
            if self.codecs[c].numeric:
                vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
            raw[c] = vals
            Zg[:, j] = cand_enc[rows, selected]
            feature_meta = {
                "feature_lambda_dp": float(lam_dp),
                "feature_lambda_eo": float(lam_eo),
                "feature_corrected_column": c,
                "candidate_count": C,
            }

        return pd.DataFrame(raw), Zg, feature_meta

    def _q_target(self, Zg: np.ndarray) -> np.ndarray:
        leaves = self.target_tree.apply(Zg)
        q = np.fromiter(
            (self.target_leaf_prob.get(int(leaf), self.global_target_p) for leaf in leaves),
            dtype=np.float64,
            count=len(leaves),
        )
        return np.clip(q, 1e-6, 1 - 1e-6)

    def generate(
        self,
        n: int,
        group_state: Dict[str, Any],
        seed: int,
        method: str = "gdt",
        k: int | None = None,
        fair: FairConfig | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        if fair is None:
            fair = FairConfig()

        kmax = len(DATASETS[self.dataset]["groups"])
        kk = kmax if k is None else max(1, min(int(k), kmax))

        feature_t0 = time.perf_counter()
        if method == "multifair" and fair.feature_correction:
            Xraw, Zg, feature_meta = self._sample_features_fair(
                n=n,
                rng=rng,
                group_state=group_state,
                k=kk,
                cfg=fair,
            )
        else:
            Xraw, Zg = self._sample_features(n, rng)
            feature_meta = {
                "feature_lambda_dp": 0.0,
                "feature_lambda_eo": 0.0,
                "feature_corrected_column": "",
                "candidate_count": int(fair.candidate_count),
            }
        feature_seconds = time.perf_counter() - feature_t0

        q = self._q_target(Zg)

        if method == "gdt":
            y = (rng.random(n) < q).astype(np.int8)
            lambdas = {"dp": 0.0, "eo": 0.0}
        else:
            grouped = apply_group_state(
                pd.concat([Xraw, pd.DataFrame({"__target__": np.zeros(n, dtype=int)})], axis=1),
                self.dataset,
                group_state,
            )
            group_cols = DATASETS[self.dataset]["groups"]
            if method == "singlefair":
                use_cols = group_cols[:1]
            else:
                use_cols = group_cols[:kk]

            # The already-fitted target CART supplies the local outcome score;
            # no separate heavy proxy model is needed.
            proxy_score = self._positive_proba(self.target_tree, Zg)
            y, lambdas = _sample_fair_targets(
                q=q,
                proxy_score=proxy_score,
                groups=grouped[use_cols],
                rng=rng,
                cfg=fair,
            )

        out = Xraw.copy()
        out["__target__"] = y.astype(int)
        out = apply_group_state(out, self.dataset, group_state)
        elapsed = time.perf_counter() - t0
        meta = {
            "fit_seconds": float(self.fit_seconds),
            "generation_seconds": float(elapsed),
            "feature_generation_seconds": float(feature_seconds),
            "samples_per_second": float(n / max(elapsed, 1e-9)),
            "final_lambda_dp": float(lambdas.get("dp", 0.0)),
            "final_lambda_eo": float(lambdas.get("eo", 0.0)),
            **feature_meta,
        }
        return out, meta


class _Counter:
    def __init__(self, codes: np.ndarray, n_groups: int, alpha: float):
        self.codes = codes
        self.G = int(n_groups)
        self.alpha = float(alpha)
        self.n = np.zeros(self.G, dtype=np.int64)
        self.pos = np.zeros(self.G, dtype=np.float64)
        self.y1_n = np.zeros(self.G, dtype=np.int64)
        self.y1_pred = np.zeros(self.G, dtype=np.float64)
        self.y0_n = np.zeros(self.G, dtype=np.int64)
        self.y0_pred = np.zeros(self.G, dtype=np.float64)

    @staticmethod
    def _rate(num, den, alpha, fallback=0.5):
        if den <= 0 and alpha <= 0:
            return fallback
        return (num + alpha) / (den + 2.0 * alpha)

    def local_costs(self, i, candidate, pred, global_state, mode="incremental"):
        g = int(self.codes[i])
        gn = self.n[g]
        gp = self.pos[g]
        N, P, y1N, y1Pred, y0N, y0Pred = global_state

        old_global_dp = self._rate(P, N, self.alpha)
        old_group_dp = self._rate(gp, gn, self.alpha, old_global_dp)
        old_dp = abs(old_group_dp - old_global_dp)

        new_global_dp = self._rate(P + candidate, N + 1, self.alpha)
        new_group_dp = self._rate(gp + candidate, gn + 1, self.alpha, new_global_dp)
        new_dp = abs(new_group_dp - new_global_dp)

        if candidate == 1:
            old_global_eo = self._rate(y1Pred, y1N, self.alpha)
            old_group_eo = self._rate(self.y1_pred[g], self.y1_n[g], self.alpha, old_global_eo)
            new_global_eo = self._rate(y1Pred + pred, y1N + 1, self.alpha)
            new_group_eo = self._rate(self.y1_pred[g] + pred, self.y1_n[g] + 1, self.alpha, new_global_eo)
        else:
            old_global_eo = self._rate(y0Pred, y0N, self.alpha)
            old_group_eo = self._rate(self.y0_pred[g], self.y0_n[g], self.alpha, old_global_eo)
            new_global_eo = self._rate(y0Pred + pred, y0N + 1, self.alpha)
            new_group_eo = self._rate(self.y0_pred[g] + pred, self.y0_n[g] + 1, self.alpha, new_global_eo)

        old_eo = abs(old_group_eo - old_global_eo)
        new_eo = abs(new_group_eo - new_global_eo)

        if mode == "absolute":
            return new_dp, new_eo
        return new_dp - old_dp, new_eo - old_eo

    def update(self, i, y, pred):
        g = int(self.codes[i])
        self.n[g] += 1
        self.pos[g] += y
        if y == 1:
            self.y1_n[g] += 1
            self.y1_pred[g] += pred
        else:
            self.y0_n[g] += 1
            self.y0_pred[g] += pred

    def violations(self, global_state):
        N, P, y1N, y1Pred, y0N, y0Pred = global_state
        valid = self.n > 0
        if not valid.any():
            return 0.0, 0.0

        a = self.alpha
        global_dp = self._rate(P, N, a)
        group_dp = (self.pos[valid] + a) / (self.n[valid] + 2 * a) if a > 0 else self.pos[valid] / self.n[valid]
        dp = float(np.mean(np.abs(group_dp - global_dp)))

        pieces = []
        v1 = self.y1_n > 0
        if v1.any():
            g_tpr = (self.y1_pred[v1] + a) / (self.y1_n[v1] + 2*a) if a > 0 else self.y1_pred[v1] / self.y1_n[v1]
            global_tpr = self._rate(y1Pred, y1N, a)
            pieces.append(np.abs(g_tpr - global_tpr))
        v0 = self.y0_n > 0
        if v0.any():
            g_fpr = (self.y0_pred[v0] + a) / (self.y0_n[v0] + 2*a) if a > 0 else self.y0_pred[v0] / self.y0_n[v0]
            global_fpr = self._rate(y0Pred, y0N, a)
            pieces.append(np.abs(g_fpr - global_fpr))
        eo = float(np.mean(np.concatenate(pieces))) if pieces else 0.0
        return dp, eo


class _FeatureCounter:
    """Online soft-prediction fairness statistics for feature-level candidates."""

    def __init__(self, n_groups: int, alpha: float):
        self.G = int(n_groups)
        self.alpha = float(alpha)
        self.n = np.zeros(self.G, dtype=np.int64)
        self.pred = np.zeros(self.G, dtype=np.float64)
        self.y1_n = np.zeros(self.G, dtype=np.int64)
        self.y1_pred = np.zeros(self.G, dtype=np.float64)
        self.y0_n = np.zeros(self.G, dtype=np.int64)
        self.y0_pred = np.zeros(self.G, dtype=np.float64)

    @staticmethod
    def _rate(num, den, alpha, fallback=0.5):
        if den <= 0 and alpha <= 0:
            return fallback
        return (num + alpha) / (den + 2.0 * alpha)

    def cost(self, g: int, yref: int, pred: float, global_state, mode="incremental"):
        N, P, y1N, y1Pred, y0N, y0Pred = global_state
        gn = self.n[g]
        gp = self.pred[g]
        a = self.alpha

        old_global_dp = self._rate(P, N, a)
        old_group_dp = self._rate(gp, gn, a, old_global_dp)
        old_dp = abs(old_group_dp - old_global_dp)
        new_global_dp = self._rate(P + pred, N + 1, a)
        new_group_dp = self._rate(gp + pred, gn + 1, a, new_global_dp)
        new_dp = abs(new_group_dp - new_global_dp)

        if yref == 1:
            old_global_eo = self._rate(y1Pred, y1N, a)
            old_group_eo = self._rate(self.y1_pred[g], self.y1_n[g], a, old_global_eo)
            new_global_eo = self._rate(y1Pred + pred, y1N + 1, a)
            new_group_eo = self._rate(self.y1_pred[g] + pred, self.y1_n[g] + 1, a, new_global_eo)
        else:
            old_global_eo = self._rate(y0Pred, y0N, a)
            old_group_eo = self._rate(self.y0_pred[g], self.y0_n[g], a, old_global_eo)
            new_global_eo = self._rate(y0Pred + pred, y0N + 1, a)
            new_group_eo = self._rate(self.y0_pred[g] + pred, self.y0_n[g] + 1, a, new_global_eo)
        old_eo = abs(old_group_eo - old_global_eo)
        new_eo = abs(new_group_eo - new_global_eo)

        if mode == "absolute":
            return new_dp, new_eo
        return new_dp - old_dp, new_eo - old_eo

    def update(self, g: int, yref: int, pred: float):
        self.n[g] += 1
        self.pred[g] += pred
        if yref == 1:
            self.y1_n[g] += 1
            self.y1_pred[g] += pred
        else:
            self.y0_n[g] += 1
            self.y0_pred[g] += pred

    def violations(self, global_state):
        N, P, y1N, y1Pred, y0N, y0Pred = global_state
        valid = self.n > 0
        if not valid.any():
            return 0.0, 0.0
        a = self.alpha
        global_dp = self._rate(P, N, a)
        group_dp = (self.pred[valid] + a) / (self.n[valid] + 2*a) if a > 0 else self.pred[valid] / self.n[valid]
        dp = float(np.mean(np.abs(group_dp - global_dp)))
        pieces = []
        v1 = self.y1_n > 0
        if v1.any():
            g_tpr = (self.y1_pred[v1] + a) / (self.y1_n[v1] + 2*a) if a > 0 else self.y1_pred[v1] / self.y1_n[v1]
            pieces.append(np.abs(g_tpr - self._rate(y1Pred, y1N, a)))
        v0 = self.y0_n > 0
        if v0.any():
            g_fpr = (self.y0_pred[v0] + a) / (self.y0_n[v0] + 2*a) if a > 0 else self.y0_pred[v0] / self.y0_n[v0]
            pieces.append(np.abs(g_fpr - self._rate(y0Pred, y0N, a)))
        eo = float(np.mean(np.concatenate(pieces))) if pieces else 0.0
        return dp, eo


def _factorize_groups(groups: pd.DataFrame, mode: str):
    work = groups.astype("string").fillna(MISSING_TOKEN)
    if mode == "joint":
        tuples = pd.MultiIndex.from_frame(work)
        codes, uniques = pd.factorize(tuples, sort=True)
        return [(codes.astype(np.int32), len(uniques))]
    result = []
    for c in work.columns:
        codes, uniques = pd.factorize(work[c], sort=True)
        result.append((codes.astype(np.int32), len(uniques)))
    return result


def _sample_fair_targets(q, proxy_score, groups, rng, cfg: FairConfig):
    factored = _factorize_groups(groups, cfg.group_mode)
    counters = [_Counter(codes, G, cfg.alpha) for codes, G in factored]

    lam_dp = float(cfg.lambda_fair) if cfg.constraint_mode in ("joint", "dp") else 0.0
    lam_eo = float(cfg.lambda_fair) if cfg.constraint_mode in ("joint", "eo") else 0.0

    y = np.zeros(len(q), dtype=np.int8)
    N = 0
    P = 0.0
    y1N = 0
    y1Pred = 0.0
    y0N = 0
    y0Pred = 0.0

    for i in range(len(q)):
        pred = float(proxy_score[i])
        global_state = (N, P, y1N, y1Pred, y0N, y0Pred)
        logw = np.empty(2, dtype=float)

        for cand in (0, 1):
            dp_cost = 0.0
            eo_cost = 0.0
            for counter in counters:
                dpc, eoc = counter.local_costs(
                    i, cand, pred, global_state, mode=cfg.penalty_mode
                )
                dp_cost += dpc
                eo_cost += eoc

            dp_cost /= max(len(counters), 1)
            eo_cost /= max(len(counters), 1)

            qy = q[i] if cand == 1 else (1.0 - q[i])
            utility_cost = -math.log(max(qy, 1e-12))
            logw[cand] = (
                math.log(max(qy, 1e-12))
                - cfg.lambda_u * utility_cost
                - lam_dp * dp_cost
                - lam_eo * eo_cost
            )

        m = float(np.max(logw))
        probs = np.exp(logw - m)
        probs /= probs.sum()
        yi = int(rng.random() < probs[1])
        y[i] = yi

        for counter in counters:
            counter.update(i, yi, pred)

        N += 1
        P += yi
        if yi == 1:
            y1N += 1
            y1Pred += pred
        else:
            y0N += 1
            y0Pred += pred

        if cfg.adapt and (i + 1) % int(cfg.update_batch) == 0:
            state = (N, P, y1N, y1Pred, y0N, y0Pred)
            vals = [c.violations(state) for c in counters]
            dp_v = float(np.mean([v[0] for v in vals]))
            eo_v = float(np.mean([v[1] for v in vals]))
            if cfg.constraint_mode in ("joint", "dp"):
                lam_dp = max(0.0, lam_dp + cfg.eta * (dp_v - cfg.epsilon))
            if cfg.constraint_mode in ("joint", "eo"):
                lam_eo = max(0.0, lam_eo + cfg.eta * (eo_v - cfg.epsilon))

    return y, {"dp": lam_dp, "eo": lam_eo}
