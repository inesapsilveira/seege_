
from time_fidelity import *
from frequency_fidelity import *
from time_frequency_fidelity import *
from complexity_fidelity import *
from spatial_fidelity import *
from diversity import *
from privacy import *

def _sim(a, b, eps=1e-12):
    """Normalised absolute-difference similarity in [0,1]."""
    return 1.0 - np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)

def as_NT(x, *, name="data"):
    """Return data as (N,T). Accepts (T,), (N,T), (N,1,T), (C,T) [single sample multi-channel not allowed]."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        # interpret as (N,T)
        return x
    if x.ndim == 3:
        # (N,C,T) -> only allowed if C==1
        if x.shape[1] != 1:
            raise ValueError(f"{name}: cannot convert multichannel (N,C,T) with C={x.shape[1]} to (N,T) without a reduction.")
        return x[:, 0, :]
    raise ValueError(f"{name}: unsupported shape {x.shape} for as_NT")

def as_NCT(x, *, name="data"):
    """Return data as (N,C,T). Accepts (C,T), (N,C,T), (N,T) [assumes C=1]."""
    x = np.asarray(x)
    if x.ndim == 2:
        # Could be (N,T) or (C,T). We disambiguate by assuming it's (N,T) and make it (N,1,T)
        # If you want to allow (C,T) single sample, pass explicitly with a flag (see below).
        return x[:, None, :]
    if x.ndim == 3:
        return x
    if x.ndim == 1:
        return x[None, None, :]
    raise ValueError(f"{name}: unsupported shape {x.shape} for as_NCT")

def _entropy_weights(values, eps=1e-12):
    """
    Compute entropy-based adaptive weights.
    Higher variability → lower entropy → higher weight.

    Parameters
    ----------
    values : list or array-like
        Vector of metric values across domains.

    Returns
    -------
    np.ndarray
        Normalized weights summing to 1.
    """
    x = np.asarray(values, dtype=float)

    # Normalize to positive probabilities
    x = np.clip(x, eps, None)
    p = x / np.sum(x)

    # Shannon entropy
    H = -np.sum(p * np.log(p + eps))
    H_max = np.log(len(p))

    # Information utility (1 - normalized entropy)
    utility = 1 - (H / H_max)

    # If all equal → fallback to uniform weights
    if not np.isfinite(utility) or utility <= 0:
        return np.ones_like(x) / len(x)

    # Weight proportional to deviation from uniformity
    w = p * utility
    w = np.clip(w, eps, None)
    return w / np.sum(w)

def _weighted_geometric_mean(x, w=None, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1.0)
    if w is None:
        w = np.ones_like(x) / x.size
    w = np.asarray(w, dtype=float)
    w = w / (np.sum(w) + eps)
    return float(np.prod(x ** w))

def _entropy_weights_matrix(values, eps=1e-12):
    """
    Compute entropy-based weights across several evaluated conditions
    (entropy weight method).
    Higher variability of a component across conditions → higher weight.

    Unlike `_entropy_weights`, which receives a single vector of scores, this
    function receives one row of scores per condition, so the weights reflect
    how much each component varies across conditions (not its value).

    Parameters
    ----------
    values : array-like, shape (n_conditions, n_components)
        Component scores for each condition evaluated against the same real
        data (e.g. RR baseline, GAN synthetic, phase-randomised surrogate,
        noise, mode collapse, memorised real).

    Returns
    -------
    np.ndarray
        Normalized weights (one per component) summing to 1.
    """
    x = np.clip(np.asarray(values, dtype=float), 0.0, None)
    n_conditions = x.shape[0]
    if n_conditions < 2:
        raise ValueError("Entropy weights need at least two conditions (rows).")

    # Normalize each component (column) to probabilities across conditions
    col_sum = x.sum(axis=0)
    p = np.divide(x, col_sum, out=np.full_like(x, 1.0 / n_conditions), where=col_sum > eps)

    # Normalized Shannon entropy per component
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    H = -plogp.sum(axis=0) / np.log(n_conditions)

    # Information utility (1 - normalized entropy)
    utility = np.clip(1.0 - H, 0.0, None)

    # If no component varies → fallback to uniform weights
    if utility.sum() <= eps:
        return np.ones(x.shape[1]) / x.shape[1]

    return utility / utility.sum()

def compute_time_fidelity_score(real_data, synthetic_data, weights=None):
    """
   Compute a time-domain fidelity score between real and synthetic signals
    using Hjorth parameter statistics.

    Integrated components (each mapped as S = 1/(1 + distance)):
        1. Normalised WD for Hjorth Activity
        2. Normalised WD for Hjorth Mobility
        3. Normalised WD for Hjorth Complexity
        4. Mahalanobis distance between Hjorth means

    All distances are in SD units:
        - Hjorth WDs are normalised by the real-data SD.
        - Mahalanobis is inherently scale-normalised via the covariance.

    Parameters
    ----------
    real_data : np.ndarray or list
        Real signals of shape [n_signals, n_samples].
    synthetic_data : np.ndarray or list
        Synthetic signals with the same shape as real_data.
    weights : dict, optional
        Weights for the components. Keys:
        {'activity','mobility','complexity','mahalanobis'}.
        Default: equal weights across all provided components 0.25 each.

    Returns
    -------
    float
        Composite time-domain fidelity score in [0, 1].
    """
    # Defaults (auto-balance across 4 components)
    if weights is None:
        w = 1.0 / 4.0
        weights = {
            'activity': w, 'mobility': w, 'complexity': w, 'mahalanobis': w
        }
    # Ensure all keys exist; missing ones default to 0 (excluded from sum)
    for k in ('activity', 'mobility', 'complexity', 'mahalanobis'):
        weights.setdefault(k, 0.0)

    # Compute base Hjorth metrics
    tf = TimeFidelity()
    hj = tf.compute_hjorth_metrics(real_data, synthetic_data, verbose=False)

    import numpy as _np

    # distance (in SD units) -> similarity
    def _sim(d):
        d = float(d)
        if not _np.isfinite(d) or d < 0:
            return 0.0
        return 1.0 / (1.0 + d)

    # Use SD-normalised WDs for Hjorth parameters
    activity_score = _sim(hj['WD_Activity_normSD'])
    mobility_score = _sim(hj['WD_Mobility_normSD'])
    complexity_score = _sim(hj['WD_Complexity_normSD'])

    # Mahalanobis is already scale-normalised
    mahalanobis_score = _sim(hj['Mahalanobis'])

    # Weighted combination
    time_fidelity_score = (
            weights['activity'] * activity_score +
            weights['mobility'] * mobility_score +
            weights['complexity'] * complexity_score +
            weights['mahalanobis'] * mahalanobis_score
    )

    # Print components
    print(f"Time Fidelity Score  : {time_fidelity_score:.3f}")
    print("Time Fidelity Components:")
    print(f"  Activity   (norm WD): {activity_score:.3f}")
    print(f"  Mobility   (norm WD): {mobility_score:.3f}")
    print(f"  Complexity (norm WD): {complexity_score:.3f}")
    print(f"  Mahalanobis        : {mahalanobis_score:.3f}")

    return time_fidelity_score

def compute_frequency_fidelity_score(real_data, synthetic_data, fs, weights=None):
    """
    Compute a similarity score between real and synthetic signals that blends
    spectral-band power, dominant frequency, coherence and a PSD–Wasserstein
    similarity term.

    Parameters
    ----------
    real_data : list | np.ndarray
        List/array of real signals (shape: [n_signals, n_samples] or 1-D).
    synthetic_data : list | np.ndarray
        List/array of synthetic signals (shape: [n_signals, n_samples] or 1-D).
    fs : int
        Sampling frequency (Hz).
    weights : dict, optional
        Weights for the four sub-scores.  Must contain the keys:
            {'relative', 'dom_freq', 'psd_coherence', 'wasserstein'}
        Defaults to: {'relative': 0.30,
                      'dom_freq': 0.30,
                      'psd_coherence': 0.30,
                      'wasserstein': 0.10}

    Example usage:
    --------------

    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_frequency_fidelity_score(real_data, synthetic_data, fs=2048)

    Returns
    -------
    float
        Composite frequency-domain fidelity score in the range (0, 1].
    """

    # Helper: robust scalar casting
    def _as_float(x, default=np.nan):
        """
        Best-effort conversion of metric outputs (float/array/dict) into a scalar float.
        """
        if isinstance(x, (int, float, np.floating)):
            return float(x)

        if isinstance(x, np.generic):
            return float(x)

        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return float(default)
            return float(np.nanmean(arr))

        if isinstance(x, dict):
            # Try common summary paths used across your metric API
            candidates = [
                ("RS Summary", "Global Mean"),
                ("RS Summary", "Mean"),
                ("Summary", "Global Mean"),
                ("Summary", "Mean"),
                ("Global Mean",),
                ("Mean",),
                ("wd",),
                ("WD",),
                ("distance",),
                ("value",),
            ]
            for path in candidates:
                v = x
                ok = True
                for k in path:
                    if isinstance(v, dict) and k in v:
                        v = v[k]
                    else:
                        ok = False
                        break
                if ok:
                    return _as_float(v, default=default)
            return float(default)

        try:
            return float(x)
        except Exception:
            return float(default)


    # Default values definition

    analysis_band = (0.5, 500.0)  # Hz
    win_seconds = 4.0            # Welch window length (s)
    window = 'hann'
    detrend = 'constant'
    overlap = 0.5                # 50%

    # Keep analysis band under Nyquist
    nyq = fs / 2.0
    analysis_band = (analysis_band[0], min(analysis_band[1], nyq))

    # Initialize FrequencyFidelity class
    frequency_fidelity = FrequencyFidelity(
        fs,
        analysis_band=analysis_band,
        win_seconds=win_seconds,
        window=window,
        detrend=detrend,
        overlap=overlap
    )


    # Weights
    if weights is None:
        weights = {
            'relative': 0.30,
            'dom_freq': 0.30,
            'psd_coherence': 0.30,
            'wasserstein': 0.10
        }

    # Fallback – add any missing keys with zero weight
    for k in ('relative', 'dom_freq', 'psd_coherence', 'wasserstein'):
        weights.setdefault(k, 0.0)

    # Optionally normalize weights to sum to 1 (prevents accidental scaling)
    wsum = float(sum(weights.values()))
    if wsum > 0:
        weights = {k: float(v) / wsum for k, v in weights.items()}


    # Shape handling
    if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
        real_data = [real_data]
    if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
        synthetic_data = [synthetic_data]


    # 1) Relative power
    freqs_r, psd_r, rel_power_r, dominant_freq_r = frequency_fidelity.compute_relative_power(
        real_data,
        analysis_band=analysis_band,
        win_seconds=win_seconds,
        window=window,
        detrend=detrend,
        overlap=overlap
    )
    freqs_s, psd_s, rel_power_s, dominant_freq_s = frequency_fidelity.compute_relative_power(
        synthetic_data,
        analysis_band=analysis_band,
        win_seconds=win_seconds,
        window=window,
        detrend=detrend,
        overlap=overlap
    )

    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    real_mean_rel_power = np.array([np.nanmean(rel_power_r[b]) for b in band_names], dtype=float)
    synth_mean_rel_power = np.array([np.nanmean(rel_power_s[b]) for b in band_names], dtype=float)

    mean_band_diff = np.nanmean(np.abs(real_mean_rel_power - synth_mean_rel_power) * 100.0)  # %
    # Map diff → similarity (clipped ≥ 0.2 to avoid 0 in extreme cases)
    relative_power_score = max(0.2, 1.0 - mean_band_diff / 20.0)
    relative_power_score = float(np.clip(relative_power_score, 0.0, 1.0))


    # 2) Dominant frequency
    freq_diff = abs(np.nanmean(dominant_freq_r) - np.nanmean(dominant_freq_s))
    dominant_freq_score = max(0.2, 1.0 - freq_diff / 3.0)
    dominant_freq_score = float(np.clip(dominant_freq_score, 0.0, 1.0))

    # 3) Coherence
    coh = frequency_fidelity.spectral_coherence(
        real_data, synthetic_data,
        mode="all_vs_all",
        per_band=False,
        analysis_band=analysis_band,
        win_seconds=2.0,
        window=window,
        detrend=detrend,
        overlap=overlap
    )

    # Robustly extract and validate coherence scalar
    coh_val = coh.get("RS Summary", {}).get("Global Mean", np.nan) if isinstance(coh, dict) else np.nan
    coh_val = _as_float(coh_val, default=np.nan)

    mean_coherence = 0.0
    if np.isfinite(coh_val):
        mean_coherence = float(np.clip(coh_val, 0.0, 1.0))

    # 4) Spectral Wasserstein distance
    wd_psd_raw = frequency_fidelity.spectral_wasserstein_distance(
        real_data, synthetic_data,
        fmin=analysis_band[0],
        fmax=analysis_band[1],
        mode="pairmean",
        per_band=False
    )

    # Robustly extract scalar WD
    wd_psd = _as_float(wd_psd_raw, default=np.nan)

    # Map distance → similarity
    wasserstein_score = 0.2
    if np.isfinite(wd_psd):
        wasserstein_score = float(1.0 / (1.0 + wd_psd))
        wasserstein_score = float(np.clip(wasserstein_score, 0.0, 1.0))

    # Composite score
    frequency_fidelity_score = (
        weights['relative']      * relative_power_score +
        weights['dom_freq']      * dominant_freq_score +
        weights['psd_coherence'] * mean_coherence +
        weights['wasserstein']   * wasserstein_score
    )

    frequency_fidelity_score = float(np.clip(frequency_fidelity_score, 0.0, 1.0))

    print(f"Frequency Fidelity Score: {frequency_fidelity_score:.2f}")

    return frequency_fidelity_score



def compute_time_frequency_fidelity_score(real_data, synthetic_data, fs, *, weights=None, mode: str = "auto",
    pad: bool = True, rr_zip_strategy: str = "consecutive", ss_zip_strategy: str = "consecutive",return_sd: bool = False,
    return_per_pair: bool = False, verbose: bool = True):
    """
    Composite time frequnecy similarity for 1-D or 2-D inputs.
    Reuses `compute_scalogram_similarity_metrics` and combines RS per-pair metrics:

        score_i = w_cssim * SSIM_i
                  + w_rmse * (1 / (1 + nRMSE_i))
                  + w_cos   * Cosine_i

    Parameters
    ----------
    real_data, synthetic_data : array_like
        1-D (T,) or 2-D (N, T). Only these are required.
    weights : dict, optional
        {'color_ssim':0.4, 'rmse':0.3, 'cosine_similarity':0.3} by default.
    mode : {"auto","zip","all_vs_all"}, default "auto"
        - "auto": use "zip" if both inputs are 1-D or both are 2-D with the same N; else "all_vs_all".
        - "zip": index-wise pairing.
        - "all_vs_all": every real vs every synthetic.
    pad : bool, default True
        Right-pad signals to a common length before CWT.
    rr_zip_strategy, ss_zip_strategy : {"consecutive","halves"}
        Zip strategies for within-set RR/SS in the metrics method.
    return_sd : bool, default False
        Also return SD of per-pair scores.
    return_per_pair : bool, default False
        Also return list of per-pair scores.
    verbose : bool, default True
        Print a brief summary.

    Example usage:
    --------------

    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_scalogram_fidelity_score(real_data, synthetic_data, fs=2048)

    Returns
    -------
    mean_score : float
    (sd_score) : float, optional if return_sd=True
    (per_pair_scores) : list[float], optional if return_per_pair=True
    """

    if weights is None:
        weights = {'color_ssim': 0.6, 'rmse': 0.2, 'cosine_similarity': 0.2}
    w_cssim = float(weights.get('color_ssim', 0.6))
    w_rmse  = float(weights.get('rmse', 0.2))
    w_cos   = float(weights.get('cosine_similarity', 0.2))

    # Normalize inputs: auto-wrap 1-D → (1, T)
    R = np.asarray(real_data, dtype=float)
    S = np.asarray(synthetic_data, dtype=float)

    if R.ndim == 1:
        R = R[None, :]  # shape (1, T)
    if S.ndim == 1:
        S = S[None, :]  # shape (1, T)

    # Auto-select pairing if requested

    if mode == "auto":
        if R.shape[0] == S.shape[0]:
            eff_mode = "zip"  # pairwise 1–1
        else:
            eff_mode = "all_vs_all"  # every real vs every synthetic
    else:
        eff_mode = mode

    scalo = TimeFrequencyFidelity(fs=fs)

    # Reuse your metrics method (does RS, RR, SS under the same mode/strategies)
    metrics = scalo.compute_scalogram_similarity_metrics(
        R, S,
        mode=eff_mode, pad=pad,
        rr_zip_strategy=rr_zip_strategy, ss_zip_strategy=ss_zip_strategy,
    )

    # RS per-pair lists → composite per-pair scores
    ssim_list  = metrics.get("Per-pair SSIM (RS)", [])
    nrmse_list = metrics.get("Per-pair NRMSE (RS)", [])
    cos_list   = metrics.get("Per-pair Cosine (RS)", [])

    scores = []
    for ssim_i, nrmse_i, cos_i in zip(ssim_list, nrmse_list, cos_list):
        cos_i = max(-1.0, min(1.0, float(cos_i)))  # safety
        sim_rmse = 1.0 / (1.0 + float(nrmse_i))    # maps to (0,1]
        score_i = w_cssim*float(ssim_i) + w_rmse*sim_rmse + w_cos*cos_i
        scores.append(float(score_i))

    if not scores:
        mean_score = float('nan'); sd_score = float('nan')
    else:
        arr = np.asarray(scores, dtype=float)
        mean_score = float(np.mean(arr))
        sd_score = float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan')

    if verbose:
        label = eff_mode if eff_mode != "zip" else f"zip"
        print(f"Time-frequency Fidelity Score: {mean_score:.3f} | mode: {label}, pairs: {len(scores)}")

    #out = (mean_score,)
    #if return_sd:
        #out += (sd_score,)
    #if return_per_pair:
        #out += (scores,)
    #return out if len(out) > 1 else out[0]


def compute_complexity_fidelity_score(
        real_data,
        synthetic_data,
        q_range=np.arange(-5, 5, 0.1),
        weights=None
    ):
    """
    Complexity fidelity score combining fractal and entropy/complexity similarities.

    Final aggregation: weighted geometric mean over available (finite) subscores.
    This penalizes collapse in any single complexity component.

    Subscores (mapped to [0,1], higher = closer match):
        - dcca, mfdfa, mfdcca, sampen, permen, lzc
    """

    # Default: equal weights across ALL subscores (6 components)
    if weights is None:
        weights = {
            'dcca': 1/6, 'mfdfa': 1/6, 'mfdcca': 1/6,
            'sampen': 1/6, 'permen': 1/6, 'lzc': 1/6
        }

    # Wrap 1D arrays
    if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
        real_data = [real_data]
    if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
        synthetic_data = [synthetic_data]

    # Helpers
    def _sim_range(a, b, lo, hi):
        d = min(abs(float(a) - float(b)), hi - lo)
        return 1.0 - d / (hi - lo)

    def _mean_Hq(signals, q_range, get_scales):
        Hqs = []
        for x in signals:
            scales = get_scales(len(x))
            if len(scales) < 4:
                continue
            _, info = nk.fractal_dfa(x, scale=scales, multifractal=True, q=q_range, show=False)
            Hq = np.asarray(info["H"])
            if np.all(np.isfinite(Hq)):
                Hqs.append(Hq)
        if not Hqs:
            return None
        return np.nanmean(np.vstack(Hqs), axis=0)


    # Fractality scores


    fs_dcca = ComplexityFidelity(real_data, synthetic_data, method='DCCA', q_range=q_range)
    fs_dcca.compute_fractal_metrics()
    H_rr, H_rs = fs_dcca.means[:2]
    F_DCCA = _sim_range(H_rr, H_rs, lo=0.3, hi=1.2)

    fs_mfdfa = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', q_range=q_range)
    fs_mfdfa.compute_fractal_metrics()
    H_r, H_s = fs_mfdfa.means
    S_H_mfdfa = _sim_range(H_r, H_s, lo=0.3, hi=1.2)

    dummy = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', q_range=q_range)
    Hq_r = _mean_Hq(real_data, q_range, dummy._get_win_sizes)
    Hq_s = _mean_Hq(synthetic_data, q_range, dummy._get_win_sizes)
    if Hq_r is not None and Hq_s is not None:
        rmse = float(np.sqrt(np.nanmean((Hq_r - Hq_s) ** 2)))
        tau = 0.15
        S_Hq = float(np.exp(-rmse / tau))
    else:
        S_Hq = np.nan

    F_MFDFA = S_H_mfdfa if np.isnan(S_Hq) else 0.5 * S_H_mfdfa + 0.5 * S_Hq

    fs_mfdcca = ComplexityFidelity(real_data, synthetic_data, method='MFDCCA', q_range=q_range)
    fs_mfdcca.compute_fractal_metrics()
    Hc_rr, Hc_rs = fs_mfdcca.means[:2]
    Da_rr, Da_rs = fs_mfdcca.deltaAlpha_means[:2]

    S_H_mfdcca = _sim_range(Hc_rr, Hc_rs, lo=0.3, hi=1.2)
    S_Dalpha = _sim_range(Da_rr, Da_rs, lo=0.0, hi=0.5)
    F_MFDCCA = 0.7 * S_H_mfdcca + 0.3 * S_Dalpha


    # Entropy / algorithmic scores

    cf_entropy = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', q_range=q_range)
    e = cf_entropy.compute_entropy_complexity_metrics(
        real_data, synthetic_data,
        sampen_m=2, sampen_r=None,
        permen_m=3, permen_tau=1,
        lzc_threshold=None,
        n_surrogates=0,
        verbose=False
    )

    def _to_sim(wd_norm):
        return np.nan if not np.isfinite(wd_norm) else 1.0 / (1.0 + float(wd_norm))

    S_SampEn = _to_sim(e.get("WD_SampEn_norm", np.nan))
    S_PermEn = _to_sim(e.get("WD_PermEn_norm", np.nan))
    S_LZC    = _to_sim(e.get("WD_LZC_norm",    np.nan))

    subscores = {
        'dcca':   F_DCCA,
        'mfdfa':  F_MFDFA,
        'mfdcca': F_MFDCCA,
        'sampen': S_SampEn,
        'permen': S_PermEn,
        'lzc':    S_LZC
    }

    # Keep only finite subscores
    keys = [k for k, v in subscores.items() if np.isfinite(v)]
    if not keys:
        raise RuntimeError("Complexity fidelity score could not be computed (all subscores NaN).")

    vals = np.array([subscores[k] for k in keys], dtype=float)

    # Collect weights for valid keys, renormalize
    w = np.array([float(weights.get(k, 0.0)) for k in keys], dtype=float)

    if float(np.sum(w)) <= 0:
        w = np.ones_like(vals) / vals.size
    else:
        w = w / float(np.sum(w))

    score = _weighted_geometric_mean(vals, w)

    # Printing
    print(f"Complexity Score: {score:0.3f}\n")
    for k, v, wi in zip(keys, vals, w):
        print(f"{k.upper():8s}: {v:0.3f}  |  weight = {wi:0.3f}")


    return float(np.clip(score, 0.0, 1.0))

def compute_spatial_fidelity_score(real_data, synthetic_data):
    """
    Compute spatial fidelity score based on cross-channel
    correlation matrix similarity.

    The score is derived from the SD-normalised deviation:

        z_SC = (μ_RS - μ_RR) / σ_RR
        F_spatial = 1 / (1 + |z_SC|)

    Parameters
    ----------
    real_data : array-like
        Shape (N, C, T) or (C, T)
    synthetic_data : array-like
        Shape (N, C, T) or (C, T)

    Returns
    -------
    float
        Spatial fidelity score in [0,1]
    """

    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)

    if real_data.ndim == 2:
        # ambiguous: could be (C,T) or (N,T)
        # spatial fidelity is only valid if axis 0 == channels
        if real_data.shape[0] < 2:
            raise ValueError("Need at least 2 channels for spatial fidelity.")
        # optional: enforce expected C
        # if real_data.shape[0] != expected_C: raise ...
    elif real_data.ndim == 3:
        if real_data.shape[1] < 2:
            raise ValueError("Need at least 2 channels for spatial fidelity.")
    else:
        raise ValueError(f"Invalid real_data shape {real_data.shape}; expected 2D or 3D.")

    spatial_eval = SpatialFidelity()
    results = spatial_eval.evaluate(real_data, synthetic_data)

    spatial_score = results.get("F_spatial", np.nan)

    # Safety clip
    spatial_score = float(np.clip(spatial_score, 0.0, 1.0))

    print(f"Spatial Fidelity Score: {spatial_score:.3f}")

    return spatial_score

use_adaptive_weights = False

def compute_fidelity_score(real_data, synthetic_data, fs, use_adaptive_weights=True,
                           spatial_requires_multichannel=True):
    """
    Computes global fidelity score using entropy-adaptive weighting
    and weighted geometric aggregation.

    Parameters
    ----------
    real_data : list or np.ndarray
        List of real signals.
    synthetic_data : list or np.ndarray
        List of synthetic signals.
    fs : int
        Sampling frequency of the signals.

    Returns
    -------
    float
        Fidelity score in [0,1] (higher ⇒ closer match).

    Example usage:
    --------------

    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_fidelity_score(real_data, synthetic_data, fs=2048)
    """
    # Univariate view for most metrics
    real_u = as_NT(real_data, name="real_data")
    syn_u = as_NT(synthetic_data, name="synthetic_data")

    time_sim = compute_time_fidelity_score(real_data, synthetic_data)
    freq_sim = compute_frequency_fidelity_score(real_data, synthetic_data, fs)
    scalogram_sim = compute_time_frequency_fidelity_score(real_data, synthetic_data, fs)
    fractal_sim = compute_complexity_fidelity_score(real_data, synthetic_data)

    # Multichannel view for spatial
    real_m = as_NCT(real_data, name="real_data")
    syn_m = as_NCT(synthetic_data, name="synthetic_data")

    # Decide what to do if C<2
    if real_m.shape[1] < 2 or syn_m.shape[1] < 2:
        if spatial_requires_multichannel:
            spatial_sim = np.nan
        else:
            # Treat as "not applicable" and don't penalize
            spatial_sim = 1.0
    else:
        spatial_sim = compute_spatial_fidelity_score(real_m, syn_m)

    components = np.array([time_sim, freq_sim, scalogram_sim, fractal_sim, spatial_sim], dtype=float)

    # Handle N/A domains (NaN) by excluding them from aggregation
    valid = np.isfinite(components)
    comps_v = components[valid]

    if comps_v.size == 0:
        return np.nan

    # weights
    if use_adaptive_weights:
        weights = _entropy_weights(comps_v)
    else:
        weights = np.ones(comps_v.size) / comps_v.size

    eps = 1e-12
    fidelity_score = np.prod(np.clip(comps_v, eps, 1.0) ** weights)
    fidelity_score = float(np.clip(fidelity_score, 0.0, 1.0))

    print(f"\nGlobal Fidelity Score: {fidelity_score:.3f}")
    print("Domain Scores:")
    print(f"  Time        : {time_sim:.3f}")
    print(f"  Frequency   : {freq_sim:.3f}")
    print(f"  Time-Freq   : {scalogram_sim:.3f}")
    print(f"  Complexity  : {fractal_sim:.3f}")
    print(f"  Spatial     : {spatial_sim if np.isfinite(spatial_sim) else float('nan'):.3f}")
    print("Weights:")
    print(f"  {weights}")

    return fidelity_score

def compute_diversity_score(real_data, synthetic_data, weights=None, n_components=2,
                            use_adaptive_weights=False, floor=0.01, return_components=False):
    """
    Composite diversity score combining coverage, geometric and intrinsic
    diversity components.

    Final aggregation: weighted geometric mean over available (finite) subscores.
    This penalizes collapse in any single diversity component.

    Subscores (mapped to [0,1], higher = closer match):
        - coverage, outliers
        - pca_labelmix, umap_labelmix, pca_overlap, umap_overlap,
          pca_covshape, umap_covshape
        - uniqueness, global_div, local_div_p10, local_div_p50
          (ratios r normalised as exp(-|ln r|))

    Parameters
    ----------
    real_data : np.ndarray
        Real signals of shape [n_signals, n_samples].
    synthetic_data : np.ndarray
        Synthetic signals of shape [n_signals, n_samples].
    weights : dict, optional
        Weights for the subscores (keys as listed above), e.g. obtained with
        `_entropy_weights_matrix` across validation conditions.
        Default: equal weights across all subscores (1/12 each).
    n_components : int, optional
        Number of PCA/UMAP components (default 2).
    use_adaptive_weights : bool, optional
        If True, use `_entropy_weights` on this single score vector (published
        behaviour; reduces to w_k = D_k / sum(D), giving the worst component the
        least weight). Default False.
    floor : float, optional
        Subscores are floored at this value before aggregation, so a single
        subscore at 0 does not force the score to 0 (default 0.01).
    return_components : bool, optional
        If True, also return the dict of subscores.

    Example usage:
    --------------

    real_data = np.random.randn(200, 2048)  # 200 real signals, each 2048 samples
    synthetic_data = np.random.randn(200, 2048) # 200 synthetic signals, each 2048 samples

    evaluation_score.compute_diversity_score(real_data, synthetic_data)

    Returns
    -------
    float (or (float, dict) if return_components=True)
        Composite diversity score in [0, 1].
    """

    # Default: equal weights across ALL subscores (12 components)
    if weights is None:
        weights = {
            'coverage': 1/12, 'outliers': 1/12,
            'pca_labelmix': 1/12, 'umap_labelmix': 1/12,
            'pca_overlap': 1/12, 'umap_overlap': 1/12,
            'pca_covshape': 1/12, 'umap_covshape': 1/12,
            'uniqueness': 1/12, 'global_div': 1/12,
            'local_div_p10': 1/12, 'local_div_p50': 1/12
        }

    # Helpers
    def _normalize_ratio(ratio):
        if ratio is None or not np.isfinite(ratio) or ratio <= 0:
            return np.nan
        return float(np.exp(-abs(np.log(ratio))))

    # Diversity metrics
    div = Diversity(n_components=n_components)
    m_cov = div.compute_coverage_diversity(real_data, synthetic_data)
    m_geom = div.compute_geometric_diversity(real_data, synthetic_data)
    m_intr = div.compute_intrinsic_diversity(real_data, synthetic_data)

    subscores = {
        'coverage':      float(m_cov['Coverage']),
        'outliers':      float(m_cov['Outliers']),
        'pca_labelmix':  float(m_geom['PCA_LabelMixingScore']),
        'umap_labelmix': float(m_geom['UMAP_LabelMixingScore']),
        'pca_overlap':   float(m_geom['PCA_OverlapMahalanobis']),
        'umap_overlap':  float(m_geom['UMAP_OverlapMahalanobis']),
        'pca_covshape':  float(m_geom['PCA_CovShape']),
        'umap_covshape': float(m_geom['UMAP_CovShape']),
        'uniqueness':    _normalize_ratio(m_intr['Uniqueness_NN']),
        'global_div':    _normalize_ratio(m_intr['Global_Diversity']),
        'local_div_p10': _normalize_ratio(m_intr['Local_Diversity_P10']),
        'local_div_p50': _normalize_ratio(m_intr['Local_Diversity_P50'])
    }

    # Keep only finite subscores
    keys = [k for k, v in subscores.items() if np.isfinite(v)]
    if not keys:
        raise RuntimeError("Diversity score could not be computed (all subscores NaN).")

    vals = np.array([subscores[k] for k in keys], dtype=float)

    # Collect weights for valid keys, renormalize
    if use_adaptive_weights:
        w = _entropy_weights(vals)
    else:
        w = np.array([float(weights.get(k, 0.0)) for k in keys], dtype=float)

    if float(np.sum(w)) <= 0:
        w = np.ones_like(vals) / vals.size
    else:
        w = w / float(np.sum(w))

    score = _weighted_geometric_mean(np.maximum(vals, floor), w)

    # Printing
    print(f"Diversity Score: {score:0.3f}\n")
    for k, v, wi in zip(keys, vals, w):
        print(f"{k.upper():14s}: {v:0.3f}  |  weight = {wi:0.3f}")

    score = float(np.clip(score, 0.0, 1.0))
    if return_components:
        return score, subscores
    return score


def compute_privacy_score(
    real_data,
    synthetic_data,
    y_real: np.ndarray | None = None,
    *,
    normalize: str | None = "zscore_global",
    length_normalize: bool = True,
    weights: dict | None = None,
) -> tuple[float, dict]:
    """
    Composite privacy score in [0, 1]; higher = safer.

    Components (all mapped to [0,1] safety scores):

      - L2 effect size d_L2      (from NN distances, R–S vs R–R)
      - DTW effect size d_DTW    (same)
      - Optional: MIR (1 - attack accuracy),
        if y_real (labels) is provided.
    """
    pr = Privacy()


    # 1) Distance-based effect sizes (Cohen-style)

    eff = pr.compute_distance_effect_sizes(
        real_data,
        synthetic_data,
        normalize=normalize,
        length_normalize=length_normalize,
    )

    d_l2  = eff["l2"]["effect_size_d"]
    d_dtw = eff["dtw"]["effect_size_d"]

    def effect_size_to_safety(d: float,
                              low: float = 0.0,
                              high: float = 0.8) -> float:
        """Map effect size d to [0,1] safety."""
        if np.isnan(d):
            return np.nan
        if d <= low:
            return 0.0
        if d >= high:
            return 1.0
        return float((d - low) / (high - low))

    scores = {
        "l2":  effect_size_to_safety(d_l2),
        "dtw": effect_size_to_safety(d_dtw),
    }

    # 2) Optional MIR component (requires true labels)

    scores["mir"] = None
    if y_real is not None:
        mir = pr.compute_mir_metrics(
            real_data,
            synthetic_data,
            y_real=y_real,
            normalize=normalize,
            verbose=False,
        )

        # You could also take max / mean of all attacks here
        attack_acc = float(mir["confidence_attack_acc"])
        attack_acc = max(0.5, min(1.0, attack_acc))  # clip to [0.5,1]
        scores["mir"] = 2.0 * (1.0 - attack_acc)


    # 3) Weights + aggregate

    base_w = {
        "l2": 1.0,
        "dtw": 1.0,
        "mir": (1.0 if scores["mir"] is not None else 0.0),
    }

    if weights is not None:
        base_w.update(weights)

    # Remove MIR weight if MIR is None
    if scores["mir"] is None:
        base_w.pop("mir", None)

    # Normalise weights
    tot = sum(base_w.values())
    base_w = {k: v / tot for k, v in base_w.items()}

    # Aggregate privacy score
    privacy_score = sum(base_w[k] * scores[k] for k in base_w)

    print(f"Privacy Score (0–1, higher = safer): {privacy_score:.2f}")
    print(f"  - L2 safety score     : {scores['l2']:.2f} (from d_L2 = {d_l2:.2f})")
    print(f"  - DTW safety score    : {scores['dtw']:.2f} (from d_DTW = {d_dtw:.2f})")
    if scores["mir"] is not None:
        print(f"  - MIR safety score    : {scores['mir']:.2f}")
    else:
        print("  - MIR safety score    : n/a (labels not provided)")

    return privacy_score, scores
