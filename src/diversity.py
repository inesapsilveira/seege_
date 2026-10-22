import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


class Diversity:
    """
    Diversity evaluation for real vs. synthetic EEG (time-series) datasets.

    Metrics (as defined in Silveira et al., Discover Neuroscience 2026, Sect. 3.2)
    ----------------------------------------------------------------------------
    1) Coverage (original feature space)
       • Coverage  (Cov): fraction of real samples with a synthetic nearest neighbour
                          inside the real sample's radius
       • Outliers  (Out): fraction of synthetic samples inside the radius of at least
                          one real sample (higher = fewer outliers)
    2) Geometric structure (PCA and UMAP embeddings)
       • Centroid overlap (O_PCA, O_UMAP)
       • Label mixing     (L_PCA, L_UMAP)
       • Covariance shape (CS_PCA, CS_UMAP)
    3) Intrinsic variability (ratios synthetic / real)
       • Uniqueness (Uni), Global diversity (GD), Local diversity (LD_P10, LD_P50)

    Corrections relative to the published version (no new metrics)
    ---------------------------------------------------------------
    * Features are z-scored with the REAL data's mean and SD before any distance, so
      a few high-amplitude features no longer dominate (`standardize=True`).
    * Coverage / Outliers: per-sample radius = distance to the k-th nearest real
      neighbour (`radius="knn"`, k=5). The single global radius
      R = 2 x median NN distance saturated at 1.0 in high dimensions.
    * Label mixing: k-NN neighbour mixing (`label_mixing="knn"`). The silhouette
      version gives mid-range values for fully separable data when a class has
      several clusters (e.g. one per subject), which real EEG usually has.
    * Centroid overlap: pooled covariance = AVERAGE of the two covariances (the
      published code summed them, inflating overlap).
    * PCA embedding fitted on REAL data only (`pca_fit="real"`), so synthetic
      samples cannot shape the axes. UMAP stays fitted on both sets: fitting on real
      and transforming synthetic pulls synthetic points onto the real manifold.
    * Real and synthetic sets are compared on equal-size subsamples (repeated,
      seeded), because NN statistics depend on sample size.
    * Plot labels use the stored n_real (the published code assumed half/half).
    * All random sampling is seeded.

    To reproduce the published values:
        Diversity(standardize=False, radius="global", label_mixing="silhouette",
                  pooled_cov="sum", pca_fit="joint", balanced=False)

    Parameters
    ----------
    n_components : int
        Dimensions of the PCA/UMAP embeddings (default 2, as in the paper).
    n_neighbors, min_dist : UMAP parameters.
    random_state : int
    max_pairs : int
        Max random pairs for mean pairwise distances.
    standardize : bool
    k : int
        Neighbours for the k-NN radius and k-NN label mixing (default 5).
    radius : {"knn", "global"}
    k_sigma : float
        Multiplier for the global radius (radius="global" only).
    label_mixing : {"knn", "silhouette"}
    pooled_cov : {"mean", "sum"}
    pca_fit : {"real", "joint"}
    balanced : bool
        Equal-size subsampling of real and synthetic sets.
    n_repeats : int
        Number of balanced subsamples when the set sizes differ.
    verbose : bool

    Example
    -------
    div = Diversity()
    cov  = div.compute_coverage_diversity(real_data, synthetic_data)
    geom = div.compute_geometric_diversity(real_data, synthetic_data)
    intr = div.compute_intrinsic_diversity(real_data, synthetic_data)
    div.plot_embeddings("UMAP", geom)
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, max_pairs=200_000,
                 standardize=True, k=5, radius="knn", k_sigma=2.0, label_mixing="knn", pooled_cov="mean",
                 pca_fit="real", balanced=True, n_repeats=5, verbose=True):
        if radius not in {"knn", "global"}:
            raise ValueError("radius must be 'knn' or 'global'.")
        if label_mixing not in {"knn", "silhouette"}:
            raise ValueError("label_mixing must be 'knn' or 'silhouette'.")
        if pooled_cov not in {"mean", "sum"}:
            raise ValueError("pooled_cov must be 'mean' or 'sum'.")
        if pca_fit not in {"real", "joint"}:
            raise ValueError("pca_fit must be 'real' or 'joint'.")
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.max_pairs = max_pairs
        self.standardize = standardize
        self.k = k
        self.radius = radius
        self.k_sigma = k_sigma
        self.label_mixing = label_mixing
        self.pooled_cov = pooled_cov
        self.pca_fit = pca_fit
        self.balanced = balanced
        self.n_repeats = n_repeats
        self.verbose = verbose

    # ------------------------------------------------------------------ helpers

    def _prep(self, real_data, synthetic_data):
        """Flatten to (n, d) and, if standardize, z-score both sets with REAL mean/SD."""
        real = np.asarray(real_data, dtype=float).reshape(len(real_data), -1)
        synth = np.asarray(synthetic_data, dtype=float).reshape(len(synthetic_data), -1)
        if real.shape[1] != synth.shape[1]:
            raise ValueError(f"Feature mismatch: real {real.shape}, synthetic {synth.shape}")
        if self.standardize:
            mu = real.mean(axis=0)
            sd = real.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            real, synth = (real - mu) / sd, (synth - mu) / sd
        return real, synth

    def _draws(self, n_real, n_syn):
        """Yield (idx_real, idx_syn): equal-size seeded subsamples, or all samples."""
        if not self.balanced or n_real == n_syn:
            yield np.arange(n_real), np.arange(n_syn)
            return
        rng = np.random.default_rng(self.random_state)
        m = min(n_real, n_syn)
        for _ in range(self.n_repeats):
            ir = np.arange(n_real) if n_real == m else np.sort(rng.choice(n_real, m, replace=False))
            is_ = np.arange(n_syn) if n_syn == m else np.sort(rng.choice(n_syn, m, replace=False))
            yield ir, is_

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------ 1) Coverage & Outliers

    def compute_coverage_diversity(self, real_data, synthetic_data, k_sigma=None):
        """
        Coverage and Outlier indices in the original (standardized) feature space.

        Coverage : fraction of real samples whose nearest synthetic neighbour lies
                   within that real sample's radius. Higher = more real modes covered.
        Outliers : fraction of synthetic samples lying within the radius of at least
                   one real sample. Higher = fewer implausible synthetic samples.

        radius="knn"    : radius_i = distance from real sample i to its k-th nearest
                          real neighbour (adapts to local density).
        radius="global" : one radius R = k_sigma x median real NN distance (published).

        Returns dict: 'Coverage', 'Outliers', 'Sigma' (median real NN distance),
        'Radius' (global R, or median per-sample radius for radius="knn").
        """
        k_sigma = self.k_sigma if k_sigma is None else k_sigma
        real, synth = self._prep(real_data, synthetic_data)
        cov, out, sig, rad = [], [], [], []

        for ir, is_ in self._draws(len(real), len(synth)):
            R, S = real[ir], synth[is_]
            D_rs = pairwise_distances(R, S)
            D_rr = pairwise_distances(R, R)
            np.fill_diagonal(D_rr, np.inf)
            nn_real = D_rr.min(axis=1)
            sigma = max(float(np.median(nn_real)), 1e-12)

            if self.radius == "knn":
                kk = min(self.k, len(R) - 1)
                radii = np.sort(D_rr, axis=1)[:, kk - 1]          # k-th real neighbour
            else:
                radii = np.full(len(R), k_sigma * sigma)

            cov.append(float(np.mean(D_rs.min(axis=1) <= radii)))
            out.append(float(np.mean((D_rs <= radii[:, None]).any(axis=0))))
            sig.append(sigma)
            rad.append(float(np.median(radii)))

        res = {"Coverage": float(np.mean(cov)), "Outliers": float(np.mean(out)),
               "Sigma": float(np.mean(sig)), "Radius": float(np.mean(rad))}

        self._log(f"[Coverage Diversity] radius: {self.radius}"
                  + (f" (k={self.k})" if self.radius == "knn" else f" (k_sigma={k_sigma})")
                  + f" | standardized={self.standardize} | median radius = {res['Radius']:.4f}")
        self._log(f"[Coverage Diversity] Coverage (real→synth): {res['Coverage']:.3f}  (↑ better)")
        self._log(f"[Coverage Diversity] Outlier Goodness (synth→real): {res['Outliers']:.3f}  (↑ better)")
        return res

    # ------------------------------------------------ 2) Geometric structure

    @staticmethod
    def _knn_mixing(X, labels, k):
        """
        Mean fraction of each point's k nearest neighbours carrying the OTHER label,
        divided by the fraction expected under perfect mixing, averaged over the two
        classes. 1 = well mixed, 0 = fully separated.
        """
        n = len(X)
        kk = min(k, n - 1)
        idx = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(X)[1][:, 1:]
        other = (labels[idx] != labels[:, None]).mean(axis=1)
        scores = []
        for c in np.unique(labels):
            m = labels == c
            expected = (n - m.sum()) / (n - 1)
            scores.append(other[m].mean() / expected)
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _geometry_metrics(self, Xr, Xs):
        """Centroid overlap, label mixing and covariance shape between two embedded sets."""
        eps = 1e-12
        Sr = np.atleast_2d(np.cov(Xr.T))
        Ss = np.atleast_2d(np.cov(Xs.T))
        Sp = (Sr + Ss) if self.pooled_cov == "sum" else 0.5 * (Sr + Ss)
        Sp = Sp + 1e-6 * np.eye(Sp.shape[0])
        delta = Xr.mean(axis=0) - Xs.mean(axis=0)
        d2 = float(delta @ np.linalg.solve(Sp, delta))
        overlap = float(np.exp(-0.5 * d2))

        X = np.vstack([Xr, Xs])
        labels = np.r_[np.zeros(len(Xr), int), np.ones(len(Xs), int)]
        if self.label_mixing == "knn":
            mixing = self._knn_mixing(X, labels, self.k)
        else:
            s = silhouette_samples(X, labels)
            mixing = float(1.0 - abs(0.5 * (s[:len(Xr)].mean() + s[len(Xr):].mean())))

        num = np.linalg.norm(Sr - Ss, ord="fro")
        den = np.linalg.norm(Sr, ord="fro") + np.linalg.norm(Ss, ord="fro") + eps
        covshape = float(np.clip(1.0 - num / den, 0.0, 1.0))
        return overlap, mixing, covshape

    def compute_geometric_diversity(self, real_data, synthetic_data):
        """
        Centroid overlap, label mixing and covariance shape in PCA and UMAP embeddings.

        PCA_/UMAP_OverlapMahalanobis ∈ (0, 1] : exp(-0.5 d²), d = Mahalanobis distance
            between centroids under the pooled (averaged) covariance.
        PCA_/UMAP_LabelMixingScore ∈ [0, 1]  : 1 = well mixed, 0 = separated.
        PCA_/UMAP_CovShape ∈ [0, 1]          : 1 - ||Σr-Σs||_F / (||Σr||_F + ||Σs||_F).

        Note: UMAP does not preserve distances or covariances, so UMAP-based values
        describe the embedding's layout, not the data's geometry; interpret them
        together with the PCA values.

        Returns the paper's keys plus 'PCA_Embedding', 'UMAP_Embedding', 'n_real'.
        """
        real, synth = self._prep(real_data, synthetic_data)
        n_real = len(real)
        combined = np.vstack([real, synth])

        pca = PCA(n_components=self.n_components, svd_solver="randomized", random_state=self.random_state)
        if self.pca_fit == "real":
            pca.fit(real)
            pca_emb = pca.transform(combined)
        else:
            pca_emb = pca.fit_transform(combined)

        umap_emb = UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors,
                        min_dist=self.min_dist, random_state=self.random_state).fit_transform(combined)

        out = {"PCA_Embedding": pca_emb, "UMAP_Embedding": umap_emb, "n_real": n_real}
        for name, emb in (("PCA", pca_emb), ("UMAP", umap_emb)):
            Er, Es = emb[:n_real], emb[n_real:]
            vals = [self._geometry_metrics(Er[ir], Es[is_]) for ir, is_ in self._draws(len(Er), len(Es))]
            o, m, c = (float(np.mean(v)) for v in zip(*vals))
            out[f"{name}_OverlapMahalanobis"] = o
            out[f"{name}_LabelMixingScore"] = m
            out[f"{name}_CovShape"] = c

        for name in ("PCA", "UMAP"):
            self._log(f"[Geometric Diversity] {name:<4} -> Label Mixing Score: {out[name + '_LabelMixingScore']:.3f}"
                      f" | Mahalanobis Overlap: {out[name + '_OverlapMahalanobis']:.3f}"
                      f" | CovShape: {out[name + '_CovShape']:.3f}")
        return out

    # ------------------------------------------------ 3) Intrinsic variability

    def compute_intrinsic_diversity(self, real_data, synthetic_data):
        """
        Within-set spread of synthetic relative to real (ratios syn/real; ≈1 ideal,
        <1 collapse, >1 over-dispersion).

        Uniqueness_NN       : mean nearest-neighbour distance ratio
        Global_Diversity    : mean pairwise distance ratio
        Local_Diversity_P10 / P50 : 10th / 50th percentile NN distance ratios
        """
        real, synth = self._prep(real_data, synthetic_data)
        rng = np.random.default_rng(self.random_state)

        def _nn(X):
            D = pairwise_distances(X)
            np.fill_diagonal(D, np.inf)
            return D.min(axis=1)

        def _mean_pairwise(X):
            n = len(X)
            if n * (n - 1) // 2 <= self.max_pairs:
                return float(pairwise_distances(X)[np.triu_indices(n, 1)].mean())
            i = rng.integers(0, n, self.max_pairs)
            j = rng.integers(0, n, self.max_pairs)
            m = i != j
            return float(np.linalg.norm(X[i[m]] - X[j[m]], axis=1).mean())

        def _ratio(a, b):
            return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 1e-12 else np.nan

        if len(real) < 2 or len(synth) < 2:
            raise ValueError("Need at least 2 real and 2 synthetic samples.")

        acc = {k: [] for k in ("Uniqueness_NN", "Global_Diversity", "Local_Diversity_P10", "Local_Diversity_P50")}
        for ir, is_ in self._draws(len(real), len(synth)):
            nr, ns = _nn(real[ir]), _nn(synth[is_])
            acc["Uniqueness_NN"].append(_ratio(ns.mean(), nr.mean()))
            acc["Global_Diversity"].append(_ratio(_mean_pairwise(synth[is_]), _mean_pairwise(real[ir])))
            acc["Local_Diversity_P10"].append(_ratio(np.percentile(ns, 10), np.percentile(nr, 10)))
            acc["Local_Diversity_P50"].append(_ratio(np.percentile(ns, 50), np.percentile(nr, 50)))

        out = {k: float(np.nanmean(v)) for k, v in acc.items()}
        self._log(f"[Intrinsic Diversity] Uniqueness (NN ratio, syn/real): {out['Uniqueness_NN']:.3f}"
                  "  (~1 ideal; <1 collapse; >1 over-dispersion)")
        self._log(f"[Intrinsic Diversity] Global Diversity (pairwise ratio, syn/real): {out['Global_Diversity']:.3f}"
                  " (~1 ideal; <1 collapse; >1 over-dispersion)")
        self._log(f"[Intrinsic Diversity] Local Diversity P10/P50 (NN ratio): {out['Local_Diversity_P10']:.3f} / "
                  f"{out['Local_Diversity_P50']:.3f} (~1 ideal; <1 collapse; >1 over-dispersion)")
        return out

    # ------------------------------------------------ 4) RR baseline

    def sanity_check(self, real_data, n_splits=3):
        """
        Run the same metrics on two random halves of the REAL data (the paper's RR
        baseline). These are the best values a perfect generator could reach at this
        sample size. If samples are grouped (subjects/sessions), split by group
        instead and call the methods on the two halves yourself.
        """
        real = np.asarray(real_data)
        rng = np.random.default_rng(self.random_state)
        verbose, self.verbose = self.verbose, False
        rows = []
        try:
            for _ in range(n_splits):
                p = rng.permutation(len(real))
                a, b = real[p[: len(p) // 2]], real[p[len(p) // 2:]]
                r = {}
                r.update(self.compute_coverage_diversity(a, b))
                g = self.compute_geometric_diversity(a, b)
                r.update({k: v for k, v in g.items() if np.isscalar(v)})
                r.update(self.compute_intrinsic_diversity(a, b))
                rows.append(r)
        finally:
            self.verbose = verbose
        df = pd.DataFrame(rows).mean(numeric_only=True)
        self._log("[Sanity check] real-vs-real baseline:\n" + df.round(3).to_string())
        return df.to_dict()

    # ------------------------------------------------ 5) Plotting

    @staticmethod
    def plot_embeddings(projection_name: str, geom: dict, save: str = None, plot_title: str = None, *,
                        kind: str = 'scatter', scatter_kws: dict | None = None, kde_kws: dict | None = None):
        """
        Plot a 2D projection (PCA or UMAP) of real vs synthetic data.
        Labels come from geom['n_real'].

        kind: 'scatter' (default) or 'kde' (filled density contours per class).
        """
        key = f"{projection_name.upper()}_Embedding"
        if key not in geom:
            raise ValueError(f"Embedding '{key}' not found. Call compute_geometric_diversity(...) first.")
        emb = np.asarray(geom[key])
        if emb.ndim != 2 or emb.shape[1] < 2:
            raise ValueError(f"'{key}' must be 2D with at least 2 columns, got shape {emb.shape}.")
        emb = emb[:, :2]

        n_total = emb.shape[0]
        n_real = geom.get("n_real")
        if n_real is None:
            raise ValueError("geom['n_real'] missing; recompute with compute_geometric_diversity.")

        df = pd.DataFrame(emb, columns=['dim1', 'dim2'])
        df['Type'] = ["Real"] * n_real + ["Synthetic"] * (n_total - n_real)

        scatter_kws = {} if scatter_kws is None else dict(scatter_kws)
        kde_kws = {} if kde_kws is None else dict(kde_kws)
        if kind not in {'scatter', 'kde'}:
            raise ValueError("kind must be either 'scatter' or 'kde'.")

        fig, ax = plt.subplots(figsize=(5, 4))
        palette = {'Real': 'limegreen', 'Synthetic': 'lightskyblue'}

        if kind == 'scatter':
            skw = dict(edgecolor='black', s=80, alpha=0.8)
            skw.update(scatter_kws)
            sns.scatterplot(data=df, x='dim1', y='dim2', hue='Type', palette=palette, legend=True, ax=ax, **skw)
            ax.legend(title='', fontsize=15)
        else:
            from matplotlib.patches import Patch
            default_kde = dict(levels=6, fill=True, thresh=0.05, alpha=0.45, bw_method='scott')
            default_kde.update(kde_kws)
            handles = []
            for name in ['Real', 'Synthetic']:
                dfi = df[df['Type'] == name]
                ok = False
                if len(dfi) >= 3:
                    try:
                        sns.kdeplot(data=dfi, x='dim1', y='dim2', color=palette[name], ax=ax, **default_kde)
                        ok = True
                    except Exception:
                        pass
                if not ok:
                    sns.scatterplot(data=dfi, x='dim1', y='dim2', color=palette[name],
                                    edgecolor='black', s=60, alpha=0.7, ax=ax)
                handles.append(Patch(facecolor=palette[name], edgecolor='k', alpha=0.45 if ok else 0.9, label=name))
            ax.legend(handles=handles, title='', fontsize=12)

        ax.set_title(plot_title if plot_title is not None else projection_name.upper(), fontsize=20)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=300)
        plt.show()
        return fig
