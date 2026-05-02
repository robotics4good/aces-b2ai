"""Configurable matrix factorisation pipeline for Bridge2AI feature analysis.

Typical usage
-------------
::

    import pandas as pd
    from aces_b2ai.analysis import FeatureAnalyzer, FeatureAnalyzerConfig

    # --- Load the dataset scalar table ---
    df = pd.read_csv("features/static_features.tsv", sep="\\t")

    # Optional: add parquet-aggregated features (EMA, PPG, MFCC, etc.)
    # via FeatureAnalyzer.aggregate_parquet() helpers (see below).

    cfg = FeatureAnalyzerConfig(
        n_components=20,
        run_nmf=True,
        run_sparse_pca=True,
        families_to_include=["f0", "voice_quality", "formants", "timing"],
    )
    analyzer = FeatureAnalyzer(cfg)
    result = analyzer.fit(df, label_col="diagnosis")

    result.plot_variance_curve(save_path="variance.png")
    result.plot_nmf_heatmap(save_path="nmf_components.png")
    result.plot_correlation_matrix(save_path="corr.png")
    result.plot_component_separation(save_path="separation.png")

    print(result.selected_features)   # list of feature names to use in model
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aces_b2ai.analysis.feature_manifest import (
    ALL_STATIC_COLUMNS,
    STATIC_FEATURE_GROUPS,
    TIMESERIES_BY_NAME,
    FeatureEntry,
    timeseries_agg_columns,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FeatureAnalyzerConfig:
    """Controls every aspect of the factorisation experiment.

    Parameters
    ----------
    families_to_include:
        Static feature group names to include from ``STATIC_FEATURE_GROUPS``.
        If empty, all groups are included.
    timeseries_to_include:
        Names of time-series feature entries (from the manifest) whose
        aggregated columns should be included when they are present in ``df``.
        If empty, only static columns are used.
    n_components:
        Number of components to compute for PCA, NMF, and SparsePCA.
    run_nmf:
        Run NMF on the non-negative subset of features.
    run_sparse_pca:
        Run SparsePCA for feature selection (slow; set to False for large N).
    sparse_pca_alpha:
        Sparsity regularisation strength for SparsePCA.
    variance_threshold:
        Minimum fraction of variance a feature must have (relative to the
        most variable feature) to pass the pre-filter.  Set to 0.0 to disable.
    correlation_threshold:
        Feature pairs with |r| above this threshold are flagged as redundant.
        The one with lower mean absolute correlation to all others is kept.
    impute_strategy:
        Imputation strategy for missing values: ``"median"``, ``"mean"``,
        or ``"zero"``.
    min_nonmissing_fraction:
        Columns with fewer than this fraction of non-missing values are
        dropped before analysis.
    random_state:
        Random seed for reproducible decompositions.
    save_dir:
        Directory for plots produced by ``FeatureAnalysisResult.plot()``.
        ``None`` → current working directory.
    """

    families_to_include: list[str] = field(default_factory=list)
    timeseries_to_include: list[str] = field(default_factory=list)
    n_components: int = 20
    run_nmf: bool = True
    run_sparse_pca: bool = False
    sparse_pca_alpha: float = 1.0
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    impute_strategy: str = "median"
    min_nonmissing_fraction: float = 0.5
    random_state: int = 42
    save_dir: str | None = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FeatureAnalysisResult:
    """All outputs from a ``FeatureAnalyzer.fit()`` call.

    Attributes
    ----------
    feature_names:
        Ordered list of feature column names after all pre-filtering.
    X_scaled:
        (n_samples, n_features) standardised feature matrix used for PCA/NMF.
    pca_variance_ratio:
        Explained variance ratio per component (from TruncatedSVD).
    pca_components:
        (n_components, n_features) principal component directions.
    pca_scores:
        (n_samples, n_components) participant scores in PCA space.
    nmf_components:
        (n_components, n_features) NMF basis matrix H.  ``None`` if NMF was
        not run or no non-negative features were available.
    nmf_scores:
        (n_samples, n_components) NMF participant activations W.
    nmf_feature_names:
        Feature names corresponding to the NMF columns.
    sparse_pca_components:
        (n_components, n_features) SparsePCA components.  ``None`` if not run.
    correlation_matrix:
        (n_features, n_features) Spearman rank-correlation matrix.
    redundant_pairs:
        List of ``(feat_a, feat_b, r)`` tuples for highly correlated pairs.
    selected_features:
        Features surviving variance filter + redundancy removal — the
        recommended input to ``DiagnosisModelConfig.scalar_feature_keys``.
    labels:
        Diagnosis label array aligned with ``X_scaled`` rows.  ``None`` if
        no label column was provided.
    n_components_90pct:
        Number of PCA components needed to explain 90 % of variance.
    cfg:
        The ``FeatureAnalyzerConfig`` used.
    """

    feature_names: list[str]
    X_scaled: np.ndarray
    pca_variance_ratio: np.ndarray
    pca_components: np.ndarray
    pca_scores: np.ndarray
    nmf_components: np.ndarray | None
    nmf_scores: np.ndarray | None
    nmf_feature_names: list[str] | None
    sparse_pca_components: np.ndarray | None
    correlation_matrix: np.ndarray
    redundant_pairs: list[tuple[str, str, float]]
    selected_features: list[str]
    labels: np.ndarray | None
    n_components_90pct: int
    cfg: FeatureAnalyzerConfig

    # ------------------------------------------------------------------
    # Convenience summaries
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise text summary of the analysis."""
        lines = [
            f"Feature matrix:       {self.X_scaled.shape[0]} samples × {self.X_scaled.shape[1]} features",
            f"Selected features:    {len(self.selected_features)}",
            f"Redundant pairs:      {len(self.redundant_pairs)}",
            f"PCA components for 90% variance: {self.n_components_90pct}",
        ]
        if self.nmf_components is not None:
            lines.append(f"NMF components:       {self.nmf_components.shape[0]} (on {len(self.nmf_feature_names)} non-neg features)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_variance_curve(self, save_path: str | None = None) -> None:
        """Cumulative explained variance curve from PCA."""
        _require_matplotlib()
        import matplotlib.pyplot as plt

        cumvar = np.cumsum(self.pca_variance_ratio)
        n = len(cumvar)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, n + 1), cumvar * 100, marker="o", ms=4)
        ax.axhline(90, color="red", ls="--", lw=1, label="90 %")
        ax.axvline(self.n_components_90pct, color="red", ls="--", lw=1)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Cumulative explained variance (%)")
        ax.set_title("PCA explained variance — effective dimensionality")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save_or_show(fig, save_path or "variance_curve.png", self.cfg.save_dir)

    def plot_correlation_matrix(
        self,
        top_n: int = 50,
        save_path: str | None = None,
    ) -> None:
        """Hierarchically-clustered Spearman correlation heatmap."""
        _require_matplotlib()
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        names = self.feature_names
        C = self.correlation_matrix

        # Cluster columns by correlation distance.
        dist = squareform(1 - np.abs(C), checks=False)
        Z = linkage(dist, method="average")
        order = dendrogram(Z, no_plot=True)["leaves"]

        # Truncate to top_n most variable for readability.
        if len(names) > top_n:
            var = self.X_scaled.var(axis=0)
            top_idx = np.argsort(var)[-top_n:]
            top_idx_sorted = [i for i in order if i in set(top_idx)]
            C_plot = C[np.ix_(top_idx_sorted, top_idx_sorted)]
            names_plot = [names[i] for i in top_idx_sorted]
        else:
            C_plot = C[np.ix_(order, order)]
            names_plot = [names[i] for i in order]

        fig, ax = plt.subplots(figsize=(min(len(names_plot) * 0.22 + 2, 20),
                                        min(len(names_plot) * 0.22 + 2, 20)))
        im = ax.imshow(C_plot, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(names_plot)))
        ax.set_yticks(range(len(names_plot)))
        ax.set_xticklabels(names_plot, rotation=90, fontsize=6)
        ax.set_yticklabels(names_plot, fontsize=6)
        ax.set_title("Spearman correlation matrix (hierarchically ordered)")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        _save_or_show(fig, save_path or "correlation_matrix.png", self.cfg.save_dir)

    def plot_nmf_heatmap(self, save_path: str | None = None) -> None:
        """Heatmap of NMF component loadings (H matrix)."""
        if self.nmf_components is None:
            warnings.warn("NMF was not run; skipping heatmap.")
            return

        _require_matplotlib()
        import matplotlib.pyplot as plt

        H = self.nmf_components        # (k, n_features)
        names = self.nmf_feature_names

        # Normalise rows so each component has max = 1 for visual clarity.
        H_norm = H / (H.max(axis=1, keepdims=True) + 1e-9)

        fig, ax = plt.subplots(figsize=(min(len(names) * 0.20 + 2, 22), H_norm.shape[0] * 0.4 + 1))
        im = ax.imshow(H_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(H_norm.shape[0]))
        ax.set_yticklabels([f"NMF-{i}" for i in range(H_norm.shape[0])], fontsize=8)
        ax.set_title("NMF component loadings — voice quality archetypes")
        fig.colorbar(im, ax=ax, shrink=0.5)
        fig.tight_layout()
        _save_or_show(fig, save_path or "nmf_heatmap.png", self.cfg.save_dir)

    def plot_component_separation(
        self,
        n_components_to_show: int = 6,
        save_path: str | None = None,
    ) -> None:
        """Box-strip plots of PCA score distributions split by diagnosis label."""
        if self.labels is None:
            warnings.warn("No labels provided; skipping separation plot.")
            return

        _require_matplotlib()
        import matplotlib.pyplot as plt

        k = min(n_components_to_show, self.pca_scores.shape[1])
        labels = self.labels
        unique_labels = sorted(set(labels))
        n_labels = len(unique_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
        color_map = dict(zip(unique_labels, colors))

        fig, axes = plt.subplots(1, k, figsize=(k * 3, 4), sharey=False)
        if k == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            scores_by_label = [self.pca_scores[labels == lbl, i] for lbl in unique_labels]
            bp = ax.boxplot(scores_by_label, patch_artist=True, notch=False,
                            medianprops={"color": "black", "lw": 2})
            for patch, lbl in zip(bp["boxes"], unique_labels):
                patch.set_facecolor(color_map[lbl])
                patch.set_alpha(0.7)

            # Overlay individual points.
            for j, (scores, lbl) in enumerate(zip(scores_by_label, unique_labels)):
                jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(scores))
                ax.scatter(j + 1 + jitter, scores, s=15, color=color_map[lbl],
                           alpha=0.7, zorder=5)

            ax.set_xticks(range(1, n_labels + 1))
            ax.set_xticklabels([str(l) for l in unique_labels], rotation=45,
                               ha="right", fontsize=7)
            ax.set_title(f"PC-{i + 1}\n({self.pca_variance_ratio[i] * 100:.1f}%)")
            ax.grid(alpha=0.3, axis="y")

        fig.suptitle("PCA score distributions by diagnosis", fontsize=11)
        fig.tight_layout()
        _save_or_show(fig, save_path or "component_separation.png", self.cfg.save_dir)

    def plot_feature_importance(
        self,
        top_n: int = 30,
        save_path: str | None = None,
    ) -> None:
        """Horizontal bar chart: features ranked by loading on first N PCA components."""
        _require_matplotlib()
        import matplotlib.pyplot as plt

        # Sum of squared loadings across first 5 PCs, weighted by variance ratio.
        k = min(5, self.pca_components.shape[0])
        weights = self.pca_variance_ratio[:k]
        importance = np.sum(
            (self.pca_components[:k] ** 2) * weights[:, None], axis=0
        )
        top_idx = np.argsort(importance)[-top_n:][::-1]
        top_names = [self.feature_names[i] for i in top_idx]
        top_imp = importance[top_idx]

        # Colour by static group family.
        group_by_col = _build_group_lookup()

        palette = {
            "f0": "#4C72B0", "loudness": "#DD8452", "voice_quality": "#55A868",
            "formants": "#C44E52", "rhythm": "#8172B2", "timing": "#937860",
            "mfcc_global": "#DA8BC3", "spectral_flux": "#8C8C8C",
            "voiced_spectral": "#CCB974", "unvoiced_spectral": "#64B5CD",
            "extended_prosody": "#B07AA1", "quality_metrics": "#76B7B2",
            "timeseries": "#E15759",
        }

        colors = [palette.get(group_by_col.get(n, "timeseries"), "#AAAAAA") for n in top_names]
        fig, ax = plt.subplots(figsize=(8, top_n * 0.28 + 1))
        bars = ax.barh(range(len(top_names)), top_imp, color=colors)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Weighted PCA loading (first 5 PCs)")
        ax.set_title(f"Top-{top_n} features by PCA importance")
        ax.grid(alpha=0.3, axis="x")

        # Legend.
        seen: set[str] = set()
        handles = []
        import matplotlib.patches as mpatches
        for n in top_names:
            grp = group_by_col.get(n, "timeseries")
            if grp not in seen:
                seen.add(grp)
                handles.append(mpatches.Patch(color=palette.get(grp, "#AAAAAA"), label=grp))
        ax.legend(handles=handles, fontsize=7, loc="lower right")
        fig.tight_layout()
        _save_or_show(fig, save_path or "feature_importance.png", self.cfg.save_dir)

    def plot(self, save_dir: str | None = None) -> None:
        """Produce all four diagnostic plots at once."""
        d = save_dir or self.cfg.save_dir
        self.plot_variance_curve(save_path=_path(d, "01_variance_curve.png"))
        self.plot_correlation_matrix(save_path=_path(d, "02_correlation_matrix.png"))
        self.plot_nmf_heatmap(save_path=_path(d, "03_nmf_heatmap.png"))
        self.plot_component_separation(save_path=_path(d, "04_component_separation.png"))
        self.plot_feature_importance(save_path=_path(d, "05_feature_importance.png"))


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------


class FeatureAnalyzer:
    """Matrix factorisation pipeline for Bridge2AI feature analysis.

    Parameters
    ----------
    cfg:
        Configuration object controlling all analysis settings.
    """

    def __init__(self, cfg: FeatureAnalyzerConfig | None = None) -> None:
        self.cfg = cfg or FeatureAnalyzerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
    ) -> FeatureAnalysisResult:
        """Run the full analysis pipeline.

        Parameters
        ----------
        df:
            DataFrame containing feature columns.  Can be ``static_features.tsv``
            directly, or a joined table that also includes aggregated time-series
            columns produced by ``aggregate_timeseries()``.
        label_col:
            Column name in ``df`` holding the diagnosis label.  Used only for
            the separation plot; not used for unsupervised decompositions.

        Returns
        -------
        FeatureAnalysisResult
        """
        cfg = self.cfg

        # 1. Select feature columns.
        feat_cols = self._select_columns(df)
        if not feat_cols:
            raise ValueError("No feature columns found in DataFrame after filtering.")

        labels = df[label_col].values if label_col and label_col in df.columns else None

        # 2. Build numeric matrix, drop low-coverage columns, impute.
        X_raw, feat_cols = self._build_matrix(df[feat_cols])

        # 3. Variance pre-filter.
        X_raw, feat_cols = self._variance_filter(X_raw, feat_cols)

        # 4. Scale.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw).astype(np.float32)

        # 5. PCA via TruncatedSVD (works on any matrix; avoids memory issues).
        pca_var, pca_comp, pca_scores = self._run_pca(X, cfg.n_components)
        n90 = int(np.searchsorted(np.cumsum(pca_var), 0.90)) + 1

        # 6. Redundancy removal.
        corr_matrix = self._spearman_correlation(X)
        redundant_pairs, selected = self._remove_redundant(
            X, feat_cols, corr_matrix, cfg.correlation_threshold
        )

        # 7. NMF on non-negative features.
        nmf_comp = nmf_scores = nmf_feat_names = None
        if cfg.run_nmf:
            nmf_comp, nmf_scores, nmf_feat_names = self._run_nmf(
                df, feat_cols, cfg.n_components, cfg.random_state
            )

        # 8. Sparse PCA (optional, slow).
        sparse_comp = None
        if cfg.run_sparse_pca:
            sparse_comp = self._run_sparse_pca(X, cfg.n_components,
                                               cfg.sparse_pca_alpha, cfg.random_state)

        return FeatureAnalysisResult(
            feature_names=feat_cols,
            X_scaled=X,
            pca_variance_ratio=pca_var,
            pca_components=pca_comp,
            pca_scores=pca_scores,
            nmf_components=nmf_comp,
            nmf_scores=nmf_scores,
            nmf_feature_names=nmf_feat_names,
            sparse_pca_components=sparse_comp,
            correlation_matrix=corr_matrix,
            redundant_pairs=redundant_pairs,
            selected_features=selected,
            labels=labels,
            n_components_90pct=n90,
            cfg=cfg,
        )

    # ------------------------------------------------------------------
    # Static helpers: parquet aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_timeseries(
        parquet_path: str | Path,
        entry: FeatureEntry,
        participant_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate one time-series parquet into per-clip scalar columns.

        Reads the parquet file, iterates rows, computes the configured
        aggregation statistics over the time axis, and returns a DataFrame
        with columns ``[participant_id, session_id, task_name, <agg_cols>...]``.

        Parameters
        ----------
        parquet_path:
            Full path to the parquet file.
        entry:
            ``FeatureEntry`` from the manifest (provides column name,
            shape, agg_stats, channel labels, and transposition flag).
        participant_cols:
            ID columns to preserve as-is.  Defaults to the standard trio.
        """
        import pyarrow.parquet as pq

        id_cols = participant_cols or ["participant_id", "session_id", "task_name"]
        table = pq.read_table(parquet_path)
        df_raw = table.to_pandas()

        agg_names = timeseries_agg_columns(entry)
        rows: list[dict] = []

        for _, row in df_raw.iterrows():
            record: dict[str, Any] = {c: row[c] for c in id_cols if c in row}
            tensor_raw = row.get(entry.column)
            if tensor_raw is None:
                rows.append(record)
                continue

            arr = _coerce_parquet_tensor(tensor_raw)

            # Ensure shape is (C, T) regardless of storage order.
            if entry.feature_type == "timeseries_1d":
                arr = arr.ravel()[None, :]   # (1, T)
            elif entry.feature_type == "timeseries_2d_tc":
                # Stored as (T, C) in the parquet (e.g. sparc_ema).
                if arr.ndim == 2 and arr.shape[1] == entry.n_channels:
                    arr = arr.T              # (T, C) → (C, T)
                elif arr.ndim == 2 and arr.shape[0] == entry.n_channels:
                    pass                     # already (C, T)
            # timeseries_2d already (C, T)

            stats = _compute_agg_stats(arr, entry.agg_stats)  # (C * n_stats,)
            record.update(dict(zip(agg_names, stats)))
            rows.append(record)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_columns(self, df: pd.DataFrame) -> list[str]:
        cfg = self.cfg
        cols: list[str] = []

        # Static families.
        families = cfg.families_to_include or list(STATIC_FEATURE_GROUPS.keys())
        for fam in families:
            for col in STATIC_FEATURE_GROUPS.get(fam, []):
                if col in df.columns:
                    cols.append(col)

        # Aggregated time-series columns (already present in df).
        for ts_name in cfg.timeseries_to_include:
            entry = TIMESERIES_BY_NAME.get(ts_name)
            if entry is None:
                warnings.warn(f"Time-series name '{ts_name}' not in manifest; skipping.")
                continue
            for col in timeseries_agg_columns(entry):
                if col in df.columns:
                    cols.append(col)

        return list(dict.fromkeys(cols))  # preserve order, deduplicate

    def _build_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        cfg = self.cfg

        # Drop columns with too many NaNs.
        threshold = int(cfg.min_nonmissing_fraction * len(df))
        df = df.dropna(axis=1, thresh=threshold)
        feat_cols = list(df.columns)

        # Convert to float.
        X = df.values.astype(np.float64)

        # Impute.
        if cfg.impute_strategy == "median":
            col_stat = np.nanmedian(X, axis=0)
        elif cfg.impute_strategy == "mean":
            col_stat = np.nanmean(X, axis=0)
        else:
            col_stat = np.zeros(X.shape[1])

        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_stat, np.where(nan_mask)[1])

        return X, feat_cols

    def _variance_filter(
        self, X: np.ndarray, cols: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        cfg = self.cfg
        if cfg.variance_threshold <= 0:
            return X, cols
        var = X.var(axis=0)
        max_var = var.max()
        if max_var == 0:
            return X, cols
        keep = var >= cfg.variance_threshold * max_var
        return X[:, keep], [c for c, k in zip(cols, keep) if k]

    @staticmethod
    def _run_pca(
        X: np.ndarray, n_components: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.decomposition import TruncatedSVD

        k = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=k, random_state=0)
        scores = svd.fit_transform(X)
        return svd.explained_variance_ratio_, svd.components_, scores

    @staticmethod
    def _spearman_correlation(X: np.ndarray) -> np.ndarray:
        """Fast Spearman correlation via rank-transform then Pearson."""
        from scipy.stats import rankdata

        R = np.apply_along_axis(rankdata, 0, X).astype(np.float32)
        # Centre ranks.
        R -= R.mean(axis=0)
        norms = np.linalg.norm(R, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        R /= norms
        return (R.T @ R).astype(np.float32)

    @staticmethod
    def _remove_redundant(
        X: np.ndarray,
        cols: list[str],
        C: np.ndarray,
        threshold: float,
    ) -> tuple[list[tuple[str, str, float]], list[str]]:
        """Greedy redundancy removal: keep the feature with lower mean |r|."""
        n = len(cols)
        # Mean absolute correlation for each feature (excluding self).
        np.fill_diagonal(C_abs := np.abs(C.copy()), 0)
        mean_corr = C_abs.mean(axis=1)

        redundant: list[tuple[str, str, float]] = []
        dropped: set[int] = set()

        for i in range(n):
            if i in dropped:
                continue
            for j in range(i + 1, n):
                if j in dropped:
                    continue
                if float(C_abs[i, j]) >= threshold:
                    # Drop the one with higher mean absolute correlation.
                    victim = i if mean_corr[i] > mean_corr[j] else j
                    redundant.append((cols[i], cols[j], float(C_abs[i, j])))
                    dropped.add(victim)

        selected = [c for k, c in enumerate(cols) if k not in dropped]
        return redundant, selected

    def _run_nmf(
        self,
        df: pd.DataFrame,
        feat_cols: list[str],
        n_components: int,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[None, None, None]:
        # Identify non-negative columns: values must all be ≥ 0 after imputation.
        nonneg_cols = []
        for col in feat_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.dropna().ge(0).all():
                    nonneg_cols.append(col)

        if len(nonneg_cols) < n_components:
            warnings.warn(
                f"Only {len(nonneg_cols)} non-negative features available; "
                f"skipping NMF (need ≥ {n_components})."
            )
            return None, None, None

        from sklearn.decomposition import NMF
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import MinMaxScaler

        X_nn = df[nonneg_cols].values.astype(np.float64)
        X_nn = SimpleImputer(strategy="median").fit_transform(X_nn)
        # NMF requires non-negative input; min-max scale to [0, 1].
        X_nn = MinMaxScaler().fit_transform(X_nn)
        X_nn = np.clip(X_nn, 0, None)

        k = min(n_components, len(nonneg_cols) - 1)
        nmf = NMF(n_components=k, random_state=random_state, max_iter=500)
        W = nmf.fit_transform(X_nn)
        H = nmf.components_

        return H, W, nonneg_cols

    def _run_sparse_pca(
        self, X: np.ndarray, n_components: int, alpha: float, random_state: int
    ) -> np.ndarray:
        from sklearn.decomposition import SparsePCA

        k = min(n_components, X.shape[1] - 1)
        spca = SparsePCA(
            n_components=k,
            alpha=alpha,
            random_state=random_state,
            n_jobs=-1,
        )
        spca.fit(X)
        return spca.components_


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------


def _coerce_parquet_tensor(raw) -> np.ndarray:
    """Convert a parquet tensor value to a proper float32 numpy array.

    Parquet stores 2D tensors as ``list<list<float>>``.  When read into
    pandas, the outer list becomes a numpy ``object`` array whose elements
    are Python lists.  ``np.asarray()`` on that gives shape ``(C,)`` with
    dtype ``object`` rather than ``(C, T)`` with dtype ``float32``.

    This helper detects that situation and uses ``np.stack`` to materialise
    the full 2D array.
    """
    arr = np.asarray(raw)
    if arr.dtype == object:
        # list-of-lists: stack inner sequences into a 2D float array.
        try:
            arr = np.stack([np.asarray(r, dtype=np.float32) for r in arr])
        except Exception:
            arr = np.array(raw, dtype=np.float32)
    else:
        arr = arr.astype(np.float32)
    return arr


def _compute_agg_stats(arr: np.ndarray, stats: list[str]) -> np.ndarray:
    """Return flat vector of (C × |stats|) scalars for array (C, T)."""
    out = []
    for stat in stats:
        if stat == "mean":
            out.append(np.nanmean(arr, axis=1))
        elif stat == "std":
            out.append(np.nanstd(arr, axis=1))
        elif stat.startswith("p"):
            pct = float(stat[1:])
            out.append(np.nanpercentile(arr, pct, axis=1))
        else:
            raise ValueError(f"Unknown aggregation stat: {stat!r}")
    # out: list of (C,) arrays → interleave as [ch0_s0, ch0_s1, ..., ch1_s0, ...]
    result = np.stack(out, axis=1).ravel()   # (C, n_stats) → (C * n_stats,)
    return result


def _build_group_lookup() -> dict[str, str]:
    """Return column-name → group-name dict for all static features."""
    lookup: dict[str, str] = {}
    for grp, cols in STATIC_FEATURE_GROUPS.items():
        for col in cols:
            lookup[col] = grp
    return lookup


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc


def _save_or_show(fig: Any, filename: str, save_dir: str | None) -> None:
    import matplotlib.pyplot as plt

    if save_dir:
        path = Path(save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _path(directory: str | None, filename: str) -> str | None:
    if directory is None:
        return None
    return str(Path(directory) / filename)


# ---------------------------------------------------------------------------
# Pre-defined analysis groups
# ---------------------------------------------------------------------------

#: Maps a group name to the feature families / time-series names it covers.
#: ``families_to_include`` → static scalar groups from ``STATIC_FEATURE_GROUPS``.
#: ``timeseries_to_include`` → time-series entry names from the manifest.
ANALYSIS_GROUPS: dict[str, dict[str, list[str]]] = {
    "sparc": {
        "families_to_include": [],
        "timeseries_to_include": [
            "sparc_loudness",
            "sparc_periodicity",
            "sparc_pitch",
            "ema",
        ],
    },
    "torchaudio": {
        "families_to_include": [],
        "timeseries_to_include": [
            "torchaudio_pitch",
            "mfcc",
            "mel",
            "spectrogram",
            "ppg",
        ],
    },
    "static": {
        "families_to_include": list(STATIC_FEATURE_GROUPS.keys()),
        "timeseries_to_include": [],
    },
}


# ---------------------------------------------------------------------------
# Multi-group result
# ---------------------------------------------------------------------------


@dataclass
class MultiGroupAnalysisResult:
    """Holds per-group ``FeatureAnalysisResult`` objects and summary stats.

    Attributes
    ----------
    groups:
        Dict mapping group name → ``FeatureAnalysisResult``.  Groups with no
        columns found in the DataFrame are absent.
    cfg:
        The base ``FeatureAnalyzerConfig`` used (before per-group overrides).
    """

    groups: dict[str, FeatureAnalysisResult]
    cfg: FeatureAnalyzerConfig

    def summary(self) -> str:
        lines = ["Multi-group factorisation summary", "=" * 40]
        for name, res in self.groups.items():
            lines.append(f"\n[{name.upper()}]")
            lines.append(res.summary())
        return "\n".join(lines)

    def plot_group_comparison(self, save_path: str | None = None) -> None:
        """Side-by-side variance curves + effective-rank bar for all groups."""
        _require_matplotlib()
        import matplotlib.pyplot as plt

        n = len(self.groups)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        if n == 1:
            axes = list(axes)

        colors = plt.cm.Set2(np.linspace(0, 1, n))
        effective_ranks: list[int] = []
        group_names: list[str] = []

        for ax, (name, res), color in zip(axes, self.groups.items(), colors):
            cumvar = np.cumsum(res.pca_variance_ratio) * 100
            ax.plot(range(1, len(cumvar) + 1), cumvar, color=color, marker="o", ms=3)
            ax.axhline(90, color="grey", ls="--", lw=1)
            ax.axvline(res.n_components_90pct, color="grey", ls="--", lw=1)
            ax.set_title(f"{name}\n({res.X_scaled.shape[1]} feats)", fontsize=9)
            ax.set_xlabel("Components")
            ax.set_ylabel("Cumul. var. (%)")
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.3)
            effective_ranks.append(res.n_components_90pct)
            group_names.append(name)

        # Rightmost panel: effective-rank bar chart.
        ax_bar = axes[-1]
        bars = ax_bar.bar(group_names, effective_ranks, color=colors[: len(group_names)])
        ax_bar.set_ylabel("Components for 90 % variance")
        ax_bar.set_title("Effective dimensionality\nper group")
        for bar, val in zip(bars, effective_ranks):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_bar.grid(alpha=0.3, axis="y")

        fig.suptitle("Feature group — PCA effective dimensionality", fontsize=11)
        fig.tight_layout()
        _save_or_show(
            fig,
            save_path or "group_comparison.png",
            self.cfg.save_dir,
        )

    def plot_all(self, save_dir: str | None = None) -> None:
        """Produce all per-group plots + the group comparison plot."""
        d = save_dir or self.cfg.save_dir
        for name, res in self.groups.items():
            sub = str(Path(d) / name) if d else None
            res.plot(save_dir=sub)
        self.plot_group_comparison(
            save_path=_path(d, "00_group_comparison.png")
        )


# ---------------------------------------------------------------------------
# Multi-group analyser
# ---------------------------------------------------------------------------


class MultiGroupFeatureAnalyzer:
    """Run independent matrix factorisation on SPARC, TorchAudio, and Static
    feature subsets.

    Parameters
    ----------
    base_cfg:
        ``FeatureAnalyzerConfig`` applied to every group.  Per-group overrides
        (``group_cfg_overrides``) take precedence.
    group_cfg_overrides:
        Dict mapping group name → partial config kwargs to override in the
        base config for that group only.  E.g.::

            {"static": {"run_sparse_pca": True, "n_components": 30}}

    groups:
        Which groups to run.  Defaults to all three: ``["sparc", "torchaudio",
        "static"]``.  Must be keys of ``ANALYSIS_GROUPS``.
    """

    def __init__(
        self,
        base_cfg: FeatureAnalyzerConfig | None = None,
        group_cfg_overrides: dict[str, dict[str, Any]] | None = None,
        groups: list[str] | None = None,
    ) -> None:
        self.base_cfg = base_cfg or FeatureAnalyzerConfig()
        self.overrides = group_cfg_overrides or {}
        self.groups = groups or list(ANALYSIS_GROUPS.keys())

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
    ) -> MultiGroupAnalysisResult:
        """Fit each group independently.

        Parameters
        ----------
        df:
            DataFrame that may contain any combination of static scalar
            columns and pre-aggregated time-series columns (produced via
            ``FeatureAnalyzer.aggregate_timeseries()``).  Groups whose
            columns are absent from ``df`` are silently skipped.
        label_col:
            Diagnosis label column (used only for separation plots).
        """
        results: dict[str, FeatureAnalysisResult] = {}

        for group_name in self.groups:
            if group_name not in ANALYSIS_GROUPS:
                warnings.warn(f"Unknown group '{group_name}'; skipping.")
                continue

            group_def = ANALYSIS_GROUPS[group_name]
            cfg = self._build_group_cfg(group_name, group_def)

            # Check whether any columns for this group are in df.
            analyzer = FeatureAnalyzer(cfg)
            candidate_cols = analyzer._select_columns(df)
            if not candidate_cols:
                warnings.warn(
                    f"Group '{group_name}': no columns found in DataFrame; skipping."
                )
                continue

            try:
                result = analyzer.fit(df, label_col=label_col)
                results[group_name] = result
            except Exception as exc:
                warnings.warn(f"Group '{group_name}' failed: {exc}")

        return MultiGroupAnalysisResult(groups=results, cfg=self.base_cfg)

    def _build_group_cfg(
        self, group_name: str, group_def: dict[str, list[str]]
    ) -> FeatureAnalyzerConfig:
        """Merge base config + group definition + per-group overrides."""
        import dataclasses

        base_fields = dataclasses.asdict(self.base_cfg)
        # Apply group definition (which families / time-series to select).
        base_fields["families_to_include"] = group_def["families_to_include"]
        base_fields["timeseries_to_include"] = group_def["timeseries_to_include"]
        # Apply per-group overrides.
        for k, v in self.overrides.get(group_name, {}).items():
            if k in base_fields:
                base_fields[k] = v
        return FeatureAnalyzerConfig(**base_fields)
