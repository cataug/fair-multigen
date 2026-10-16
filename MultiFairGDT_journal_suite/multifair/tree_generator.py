from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple
import math
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

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
    constraint_mode: str = "joint"  # joint, dp, eo
    adapt: bool = True
    group_mode: str = "joint"       # joint, marginal
    penalty_mode: str = "incremental"  # incremental, absolute


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
        s = pd.Series(values)
        if self.numeric:
            return pd.to_numeric(s, errors="coerce").fillna(self.median).to_numpy(dtype=np.float32)
        ss = s.astype("string").fillna(MISSING_TOKEN).astype(str)
        return ss.map(lambda v: self.mapping.get(v, -1)).to_numpy(dtype=np.float32)


class AutoRegressiveTreeGenerator:
    """
    Self-contained autoregressive CART generator.

    Each feature z_t is modelled by a CART partition conditioned on z_<t.
    Generation routes a partial synthetic row to a leaf and samples an
    empirical training value from that leaf. The binary target is generated
    last from a target CART; fairness-aware methods reweight its two candidate
    probabilities using online group statistics.
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
        self.target_tree = None
        self.target_leaf_prob: Dict[int, float] = {}
        self.proxy = None
        self.n_train = 0
        self.fit_seconds = 0.0

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
        self.target_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed + 999,
        )
        self.target_tree.fit(Z, y)
        target_leaves = self.target_tree.apply(Z)
        self.target_leaf_prob = {}
        global_p = float(np.mean(y))
        for leaf in np.unique(target_leaves):
            yy = y[target_leaves == leaf]
            # Jeffreys smoothing keeps q away from exact 0/1.
            self.target_leaf_prob[int(leaf)] = float((yy.sum() + 0.5) / (len(yy) + 1.0))
        self.global_target_p = global_p

        # Fast surrogate predictor used only inside the generator's fairness
        # correction. Downstream evaluation uses separate classifiers.
        self.proxy = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=31,
            learning_rate=0.08,
            l2_regularization=1e-3,
            random_state=self.seed,
        )
        self.proxy.fit(Z, y)
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

                # Recover numeric dtype where possible.
                if self.codecs[c].numeric:
                    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()

            raw[c] = vals
            Zg[:, j] = self.codecs[c].encode(vals)

        return pd.DataFrame(raw), Zg

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
        Xraw, Zg = self._sample_features(n, rng)
        q = self._q_target(Zg)

        if method == "gdt":
            y = (rng.random(n) < q).astype(np.int8)
            lambdas = {"dp": 0.0, "eo": 0.0}
        else:
            if fair is None:
                fair = FairConfig()
            grouped = apply_group_state(
                pd.concat([Xraw, pd.DataFrame({"__target__": np.zeros(n, dtype=int)})], axis=1),
                self.dataset,
                group_state,
            )
            group_cols = DATASETS[self.dataset]["groups"]
            if method == "singlefair":
                use_cols = group_cols[:1]
            else:
                kk = len(group_cols) if k is None else int(k)
                use_cols = group_cols[:kk]

            proxy_score = self.proxy.predict_proba(Zg)[:, 1]
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
        meta = {
            "fit_seconds": float(self.fit_seconds),
            "generation_seconds": float(time.perf_counter() - t0),
            "samples_per_second": float(n / max(time.perf_counter() - t0, 1e-9)),
            "final_lambda_dp": float(lambdas.get("dp", 0.0)),
            "final_lambda_eo": float(lambdas.get("eo", 0.0)),
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


def _factorize_groups(groups: pd.DataFrame, mode: str):
    work = groups.astype("string").fillna("__MISSING__")
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
