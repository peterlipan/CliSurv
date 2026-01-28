import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict


class SurvivalDataGenerator:
    """Base class for survival data simulation with censoring."""

    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = n
        self.d = d
        self.rng = np.random.default_rng(seed)

    def _make_beta(self, n_nonzero: int = 10, scale: float = 0.5) -> np.ndarray:
        beta = np.zeros(self.d, dtype=float)
        idx = self.rng.choice(self.d, size=min(n_nonzero, self.d), replace=False)
        beta[idx] = self.rng.normal(loc=0.0, scale=scale, size=len(idx))
        return beta

    def _make_covariates(
        self,
        x_dist: str = "normal",
        corr: float = 0.0,
        heavy_tail_df: Optional[float] = None,
        mix_prob: float = 0.0,
        mix_shift: float = 2.0,
    ) -> np.ndarray:
        """
        Generate covariates with various complexity options.
        
        Args:
            x_dist: "normal" or "uniform"
            corr: AR(1) correlation coefficient in [0,1)
            heavy_tail_df: Student-t degrees of freedom for heavy tails
            mix_prob: Probability of shifted component for multi-modality
            mix_shift: Shift magnitude for mixture component
        """
        if corr < 0.0 or corr >= 1.0:
            raise ValueError("corr must be in [0, 1).")

        if heavy_tail_df is not None:
            X0 = self.rng.standard_t(df=heavy_tail_df, size=(self.n, self.d))
        else:
            if x_dist == "normal":
                X0 = self.rng.normal(size=(self.n, self.d))
            elif x_dist == "uniform":
                X0 = self.rng.uniform(low=-1.0, high=1.0, size=(self.n, self.d))
            else:
                raise ValueError("x_dist must be 'normal' or 'uniform'")

        if mix_prob > 0.0:
            mask = self.rng.uniform(size=(self.n, 1)) < mix_prob
            X0 = X0 + mask * mix_shift

        if corr > 0.0:
            eps = X0
            X = np.empty_like(eps)
            X[:, 0] = eps[:, 0]
            scale = np.sqrt(max(1e-8, 1.0 - corr**2))
            for j in range(1, self.d):
                X[:, j] = corr * X[:, j - 1] + scale * eps[:, j]
            return X.astype(float)

        return X0.astype(float)

    def _apply_censoring_mixture(
        self,
        T: np.ndarray,
        target_rate: float,
        admin_frac: float = 0.5,
        admin_time_quantile: float = 0.85,
        max_iter: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Apply non-informative censoring via mixture of administrative and exponential.
        """
        admin_frac = float(np.clip(admin_frac, 0.0, 1.0))
        target_rate = float(np.clip(target_rate, 0.0, 0.99))

        A = float(np.quantile(T, admin_time_quantile))
        is_admin = self.rng.uniform(size=len(T)) < admin_frac
        C_admin = np.full_like(T, fill_value=A, dtype=float)

        lo, hi = -12.0, 6.0
        best_rate = 1.0
        best_err = float("inf")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            rate = np.exp(mid)
            C_exp_test = self.rng.exponential(scale=1.0 / rate, size=len(T))

            C_test = np.where(is_admin, C_admin, C_exp_test)
            censor_frac = np.mean(C_test < T)
            err = abs(censor_frac - target_rate)

            if err < best_err:
                best_err = err
                best_rate = rate

            if censor_frac < target_rate:
                lo = mid
            else:
                hi = mid

        C_exp = self.rng.exponential(scale=1.0 / best_rate, size=len(T))
        C = np.where(is_admin, C_admin, C_exp)

        Y = np.minimum(T, C)
        E = (T <= C).astype(int)

        info = {
            "target_censor": target_rate,
            "achieved_censor": float(np.mean(E == 0)),
            "admin_time": A,
            "admin_frac": admin_frac,
            "exp_rate": float(best_rate),
        }
        return Y, E, info

    def _make_dataframe(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        E: np.ndarray,
        extra_cols: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(self.d)])
        df["duration"] = Y
        df["event"] = E
        if extra_cols:
            for name, values in extra_cols.items():
                df[name] = values
        return df

    # -----------------------------
    # Nonlinear risk generator
    # -----------------------------
    def _make_risk(
        self,
        X: np.ndarray,
        beta: Optional[np.ndarray] = None,
        n_nonzero: int = 10,
        beta_scale: float = 0.3,
        # nonlinear controls
        nonlinear: bool = True,
        lin_weight: float = 0.35,     # weight of linear part in total risk
        nl_weight: float = 0.65,      # weight of nonlinear part in total risk
        n_spline_feats: int = 24,     # number of features that get nonlinear transforms
        n_interactions: int = 32,     # number of pairwise interactions
        n_thresholds: int = 16,       # number of threshold/hinge features
        n_fourier: int = 12,          # sinusoidal features
        hidden_dim: int = 32,         # random feature map size (like a fixed 1-layer net)
        rf_scale: float = 0.7,        # scale of random-feature weights
        out_scale: float = 0.9,       # overall scale before standardizing
        standardize: bool = True,     # standardize risk to mean 0, std 1
        return_components: bool = False,
    ):
        """
        Create a harder-to-linear-fit log-risk r(x), while keeping the same PH/PO/Gen link forms.

        r(x) = lin_weight * (X @ beta) + nl_weight * f_nonlinear(X)  (if nonlinear=True)

        f_nonlinear includes:
          - smooth nonlinear transforms (tanh/softplus-like)
          - sparse pairwise interactions
          - threshold/hinge features
          - sinusoidal (Fourier) features
          - random feature map (fixed random 1-hidden-layer features)
        """
        n, d = X.shape

        if beta is None:
            beta = self._make_beta(n_nonzero=n_nonzero, scale=beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.shape[0] != d:
            raise ValueError(f"beta must have shape ({d},), got {beta.shape}")

        # ----- linear part -----
        r_lin = X @ beta

        if not nonlinear:
            r = r_lin
            if standardize:
                r = (r - r.mean()) / (r.std() + 1e-8)
            return (r, {"beta": beta, "r_lin": r_lin}) if return_components else r

        # helper: stable softplus
        def _softplus(z):
            z = np.clip(z, -50, 50)
            return np.log1p(np.exp(z))

        # ----- nonlinear transforms on selected coordinates -----
        idx_nl = self.rng.choice(d, size=min(n_spline_feats, d), replace=False)
        Xnl = X[:, idx_nl]

        # random slopes + shifts to make many different smooth nonlinearities
        a = self.rng.normal(0.0, 1.0, size=(Xnl.shape[1],))
        b = self.rng.normal(0.0, 0.7, size=(Xnl.shape[1],))
        # mix tanh and softplus-ish transforms
        phi1 = np.tanh(1.2 * (Xnl * a[None, :] + b[None, :]))
        phi2 = _softplus(0.9 * (Xnl - b[None, :])) - _softplus(-0.9 * b[None, :])
        nl_smooth = np.concatenate([phi1, phi2], axis=1)  # (n, 2*|idx_nl|)

        w_smooth = self.rng.normal(0.0, 1.0 / np.sqrt(nl_smooth.shape[1]), size=(nl_smooth.shape[1],))
        r_smooth = nl_smooth @ w_smooth

        # ----- sparse interactions -----
        # pick pairs (i,j) and multiply Xi * Xj
        pairs_i = self.rng.integers(0, d, size=n_interactions)
        pairs_j = self.rng.integers(0, d, size=n_interactions)
        # avoid i=j too often
        mask_eq = pairs_i == pairs_j
        pairs_j[mask_eq] = (pairs_j[mask_eq] + 1) % d

        inter = X[:, pairs_i] * X[:, pairs_j]  # (n, n_interactions)
        w_inter = self.rng.normal(0.0, 1.0 / np.sqrt(n_interactions), size=(n_interactions,))
        r_inter = inter @ w_inter

        # ----- threshold / hinge features -----
        thr_idx = self.rng.choice(d, size=min(n_thresholds, d), replace=False)
        thr = self.rng.normal(0.0, 0.7, size=(len(thr_idx),))
        hinge = np.maximum(0.0, X[:, thr_idx] - thr[None, :])  # (n, n_thresholds)
        w_hinge = self.rng.normal(0.0, 1.0 / np.sqrt(hinge.shape[1]), size=(hinge.shape[1],))
        r_hinge = hinge @ w_hinge

        # ----- Fourier-ish features -----
        four_idx = self.rng.choice(d, size=min(n_fourier, d), replace=False)
        freq = self.rng.uniform(0.6, 2.0, size=(len(four_idx),))
        phase = self.rng.uniform(-np.pi, np.pi, size=(len(four_idx),))
        four = np.sin(freq[None, :] * X[:, four_idx] + phase[None, :])  # (n, n_fourier)
        w_four = self.rng.normal(0.0, 1.0 / np.sqrt(four.shape[1]), size=(four.shape[1],))
        r_four = four @ w_four

        # ----- random feature map (fixed random 1-layer net) -----
        # this makes the mapping more "MLP-like" without changing the PH/PO/Gen link definitions
        W = self.rng.normal(0.0, rf_scale / np.sqrt(d), size=(d, hidden_dim))
        b0 = self.rng.normal(0.0, 0.4, size=(hidden_dim,))
        H = np.tanh(X @ W + b0[None, :])  # (n, hidden_dim)
        w_rf = self.rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim,))
        r_rf = H @ w_rf

        # ----- combine nonlinear pieces -----
        r_nl = (0.40 * r_smooth + 0.25 * r_inter + 0.15 * r_hinge + 0.10 * r_four + 0.10 * r_rf)
        r = out_scale * (lin_weight * r_lin + nl_weight * r_nl)

        if standardize:
            r = (r - r.mean()) / (r.std() + 1e-8)

        if return_components:
            return r, {
                "beta": beta,
                "r_lin": r_lin,
                "r_nl": r_nl,
                "r_smooth": r_smooth,
                "r_inter": r_inter,
                "r_hinge": r_hinge,
                "r_four": r_four,
                "r_rf": r_rf,
                "idx_nl": idx_nl,
            }
        return r



def _invert_piecewise_cumhaz(
    u: np.ndarray,
    r: np.ndarray,
    breaks: np.ndarray,
    hazards: np.ndarray,
) -> np.ndarray:
    """
    Invert cumulative hazard for PH with piecewise-constant baseline hazard.
    """
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    target = (-np.log(u) / np.exp(r)).astype(float)

    K = len(hazards)
    seg_len = np.diff(breaks)
    seg_cum = np.concatenate([[0.0], np.cumsum(hazards * seg_len)])

    T = np.empty_like(target, dtype=float)
    for i, z in enumerate(target):
        k = int(np.searchsorted(seg_cum, z, side="right") - 1)
        k = max(0, min(K - 1, k))
        z0 = seg_cum[k]
        dt = (z - z0) / max(hazards[k], 1e-12)
        t = breaks[k] + dt

        if z >= seg_cum[-1]:
            z_tail = z - seg_cum[-1]
            t = breaks[-1] + z_tail / max(hazards[-1], 1e-12)

        T[i] = t
    return T


class ProportionalHazardsPiecewiseSimulator(SurvivalDataGenerator):
    """PH model with piecewise-constant baseline hazard."""

    def generate(
        self,
        beta: Optional[np.ndarray] = None,
        n_nonzero: int = 10,
        beta_scale: float = 0.3,
        x_dist: str = "normal",
        corr: float = 0.5,
        heavy_tail_df: Optional[float] = 4.0,
        mix_prob: float = 0.2,
        mix_shift: float = 1.5,
        n_knots: int = 6,
        t_max: float = 99.0,
        hazard_level: float = 0.06,
        hazard_jitter: float = 0.8,
        target_censor: float = 0.6,
        censor_admin_frac: float = 0.5,
        censor_admin_q: float = 0.85,
    ) -> pd.DataFrame:

        X = self._make_covariates(
            x_dist=x_dist,
            corr=corr,
            heavy_tail_df=heavy_tail_df,
            mix_prob=mix_prob,
            mix_shift=mix_shift,
        )

        if beta is None:
            beta = self._make_beta(n_nonzero, beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        assert len(beta) == self.d

        r = self._make_risk(
            X,
            beta=beta,
            n_nonzero=n_nonzero,
            beta_scale=beta_scale,
            nonlinear=True,
            lin_weight=0.1,
            nl_weight=0.9,
            n_spline_feats=48,
            n_interactions=80,
            n_thresholds=40,
            n_fourier=14,
            hidden_dim=128,
            rf_scale=0.8,
            out_scale=1.0,
            standardize=True,
        )


        internal = np.linspace(0, t_max, n_knots + 1)[1:-1]
        breaks = np.concatenate([[0.0], internal, [t_max]])
        K = len(breaks) - 1

        hazards = hazard_level * self.rng.lognormal(mean=0.0, sigma=hazard_jitter, size=K)

        U = self.rng.uniform(size=self.n)
        T = _invert_piecewise_cumhaz(U, r, breaks, hazards)

        Y, E, info = self._apply_censoring_mixture(
            T, target_rate=target_censor, admin_frac=censor_admin_frac, admin_time_quantile=censor_admin_q
        )

        # cumulative baseline hazard at right endpoints
        seg_len = np.diff(breaks)
        oracle_cum_hazard = np.cumsum(hazards * seg_len)
        oracle_survival = np.exp(-oracle_cum_hazard)

        extra = {
            "true_risk": r.astype(np.float32),
            
        }
        oracle = {
            "link": "PH",
            "hazards": hazards.astype(np.float32),
            "breaks": breaks.astype(np.float32),
            "cum_hazard": oracle_cum_hazard.astype(np.float32),
            "survival": oracle_survival.astype(np.float32),
        }

        return self._make_dataframe(X, Y, E, extra_cols=extra), oracle


class ProportionalOddsPiecewiseSimulator(SurvivalDataGenerator):
    """PO model with piecewise-constant baseline odds."""

    def generate(
        self,
        beta: Optional[np.ndarray] = None,
        n_nonzero: int = 10,
        beta_scale: float = 0.3,
        x_dist: str = "normal",
        corr: float = 0.5,
        heavy_tail_df: Optional[float] = 4.0,
        mix_prob: float = 0.2,
        mix_shift: float = 1.5,
        n_knots: int = 6,
        t_max: float = 20.0,
        odds_level: float = 0.15,
        odds_jitter: float = 0.9,
        target_censor: float = 0.6,
        censor_admin_frac: float = 0.5,
        censor_admin_q: float = 0.85,
    ) -> pd.DataFrame:

        X = self._make_covariates(
            x_dist=x_dist,
            corr=corr,
            heavy_tail_df=heavy_tail_df,
            mix_prob=mix_prob,
            mix_shift=mix_shift,
        )

        if beta is None:
            beta = self._make_beta(n_nonzero, beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        assert len(beta) == self.d

        r = self._make_risk(
            X,
            beta=beta,
            n_nonzero=n_nonzero,
            beta_scale=beta_scale,
            nonlinear=True,
            lin_weight=0.30,
            nl_weight=0.70,
            n_spline_feats=24,
            n_interactions=40,
            n_thresholds=20,
            n_fourier=14,
            hidden_dim=40,
            rf_scale=0.8,
            out_scale=1.0,
            standardize=True,
        )
        er = np.exp(r)


        internal = np.sort(self.rng.uniform(low=0.5, high=t_max, size=max(0, n_knots - 1)))
        breaks = np.concatenate([[0.0], internal, [t_max]])
        K = len(breaks) - 1

        # --- NEW: incremental cumulative odds (PH-style) ---
        delta = odds_level * self.rng.lognormal(
            mean=0.0,
            sigma=odds_jitter,
            size=K
        )  # small positive increments

        time_frac = np.linspace(0.0, 1.0, K)

        # PO-specific time warping (concave → convex)
        shape = 1.8        # >1 = slow early, fast late
        weights = time_frac ** shape

        odds0 = np.cumsum(delta * weights)

        # normalize tail
        odds_max = 5.0
        odds0 = odds0 / odds0[-1] * odds_max


        # optional: control final survival level
        odds_max = 5.0   # S0_end ≈ 1 / (1 + 5) ≈ 0.17
        odds0 = odds0 / odds0[-1] * odds_max

        U = self.rng.uniform(size=self.n)
        odds_target = np.clip(U / (1.0 - U), 1e-12, 1e12)
        odds0_target = odds_target / np.clip(er, 1e-12, None)

        idx = np.searchsorted(odds0, odds0_target, side="left")
        idx = np.clip(idx, 0, K - 1)

        u2 = self.rng.uniform(size=self.n)
        T = breaks[idx] + u2 * (breaks[idx + 1] - breaks[idx])

        Y, E, info = self._apply_censoring_mixture(
            T, target_rate=target_censor, admin_frac=censor_admin_frac, admin_time_quantile=censor_admin_q
        )

        oracle_survival = 1.0 / (1.0 + odds0)

        extra = {
            "true_risk": r.astype(np.float32),
        }
        oracle = {
            "link": "PO",
            "odds": odds0.astype(np.float32),
            "breaks": breaks.astype(np.float32),
            "survival": oracle_survival.astype(np.float32),
        }

        return self._make_dataframe(X, Y, E, extra_cols=extra), oracle


class StochasticMonotoneLink:
    """Random monotone inverse-link function."""

    def __init__(
        self,
        z_min: float = -8.0,
        z_max: float = 8.0,
        L_star: int = 24,
        alpha_range: Tuple[float, float] = (0.3, 2.0),
        w_scale: float = 1.2,
        slope_range: Tuple[float, float] = (0.5, 3.0),
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.z_min = z_min
        self.z_max = z_max

        self.b = rng.normal(0.0, 0.7)
        self.alpha = rng.uniform(*alpha_range)

        self.t = np.sort(rng.uniform(z_min, z_max, size=L_star))
        self.w = rng.lognormal(mean=np.log(max(w_scale, 1e-6)), sigma=0.8, size=L_star)
        self.s = rng.uniform(*slope_range, size=L_star)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return np.log1p(np.exp(x))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def phi(self, z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, self.z_min, self.z_max)
        ramps = self._softplus(self.s[None, :] * (zc[..., None] - self.t[None, :]))
        return self.b + self.alpha * zc + ramps @ self.w

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.phi(z))


class DiscreteTimeLinkSimulatorHard(SurvivalDataGenerator):
    """Discrete-time model with complex monotone link function."""

    def generate(
        self,
        m: int = 60,
        tau_max: float = 20.0,
        taus: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        n_nonzero: int = 10,
        beta_scale: float = 0.35,
        x_dist: str = "normal",
        corr: float = 0.5,
        heavy_tail_df: Optional[float] = 4.0,
        mix_prob: float = 0.2,
        mix_shift: float = 1.5,
        target_censor: float = 0.6,
        censor_admin_frac: float = 0.5,
        censor_admin_q: float = 0.85,
        z_min: float = -8.0,
        z_max: float = 8.0,
        L_star: int = 24,
        eta_warp: float = 2.0,
        eta_bumps: int = 3,
    ) -> pd.DataFrame:

        if taus is None:
            u = np.linspace(0.0, 1.0, m + 1)
            u = u**1.8
            taus = tau_max * u
        else:
            taus = np.asarray(taus, dtype=float)
            m = len(taus) - 1
        assert np.all(np.diff(taus) > 0), "taus must be strictly increasing"

        X = self._make_covariates(
            x_dist=x_dist,
            corr=corr,
            heavy_tail_df=heavy_tail_df,
            mix_prob=mix_prob,
            mix_shift=mix_shift,
        )

        if beta is None:
            beta = self._make_beta(n_nonzero, beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        assert len(beta) == self.d

        r = self._make_risk(
            X,
            beta=beta,
            n_nonzero=n_nonzero,
            beta_scale=beta_scale,
            nonlinear=True,
            lin_weight=0.25,
            nl_weight=0.75,
            n_spline_feats=28,
            n_interactions=48,
            n_thresholds=24,
            n_fourier=16,
            hidden_dim=48,
            rf_scale=0.9,
            out_scale=1.0,
            standardize=True,
        )


        t = np.linspace(1.0 / (m + 2), (m + 1.0) / (m + 2), m)
        eta = np.log(t / (1.0 - t))

        eta = eta_warp * eta
        centers = self.rng.uniform(z_min * 0.6, z_max * 0.6, size=eta_bumps)
        widths = self.rng.uniform(0.6, 1.8, size=eta_bumps)
        amps = self.rng.normal(0.0, 0.7, size=eta_bumps)
        for c, w, a in zip(centers, widths, amps):
            eta = eta + a * np.tanh((eta - c) / w)

        eta = (eta - eta.mean()) / (eta.std() + 1e-8)
        eta = np.clip(2.2 * eta, z_min, z_max)

        link = StochasticMonotoneLink(
            z_min=z_min, z_max=z_max, L_star=L_star, seed=int(self.rng.integers(1e9))
        )

        Z = r[:, None] + eta[None, :]
        F = link(Z)
        F = np.maximum.accumulate(F, axis=1)
        F = np.clip(F, 1e-8, 1.0 - 1e-8)

        F_prev = np.concatenate([np.zeros((self.n, 1)), F[:, :-1]], axis=1)
        pmf_bins = np.clip(F - F_prev, 1e-12, None)
        tail = np.clip(1.0 - F[:, -1], 1e-12, None)
        pmf = np.concatenate([pmf_bins, tail[:, None]], axis=1)
        pmf /= pmf.sum(axis=1, keepdims=True)

        K = np.array([self.rng.choice(m + 1, p=pmf[i]) for i in range(self.n)]) + 1

        T = np.empty(self.n, dtype=float)
        for i in range(self.n):
            k = K[i]
            if k <= m:
                T[i] = self.rng.uniform(taus[k - 1], taus[k])
            else:
                scale = max(1e-6, (taus[-1] - taus[-2]) * 1.5)
                T[i] = taus[-1] + self.rng.exponential(scale=scale)

        Y, E, info = self._apply_censoring_mixture(
            T, target_rate=target_censor, admin_frac=censor_admin_frac, admin_time_quantile=censor_admin_q
        )

        # baseline survival for r = 0 (true baseline; does NOT depend on sample risk)
        Z0 = (0.0 + eta)[None, :]              # shape (1, m)
        F0 = link(Z0)[0]                       # shape (m,)
        F0 = np.maximum.accumulate(F0)
        F0 = np.clip(F0, 1e-8, 1.0 - 1e-8)
        oracle_survival = 1.0 - F0

        # save oracle link on a grid (for link recovery plots)
        z_grid = np.linspace(z_min, z_max, 400)
        oracle_link = link(z_grid)

        extra = {
            "true_risk": r.astype(np.float32),
        }
        oracle = {
            "link": "Gen",
            "taus": taus.astype(np.float32),
            "eta": eta.astype(np.float32),
            "survival": oracle_survival.astype(np.float32),
            "link_z": z_grid.astype(np.float32),
            "link_values": oracle_link.astype(np.float32),
        }


        return self._make_dataframe(X, Y, E, extra_cols=extra), oracle


class SurvivalPyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for survival data."""
    
    def __init__(self, data: np.ndarray, duration: np.ndarray, event: np.ndarray, 
                 label: np.ndarray, n_features: int, n_classes: int, n_events: int, 
                 duration_to_label, survival_at_grids):
        self.data = data.astype(np.float32)
        self.duration = duration.astype(np.float32)
        self.event = event.astype(np.int64)
        self.label = label.astype(np.int64)
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_events = n_events
        self._duration_to_label = duration_to_label
        self.survival_at_grids = survival_at_grids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.label[idx],
            "duration": self.duration[idx],
            "event": self.event[idx],
        }


class BaseSurvivalDataset:
    """Base class for generating, caching, and loading survival datasets."""
    
    SIMULATOR_CLASS = None
    SIMULATOR_TYPE = None
    DEFAULT_PARAMS = {}
    TIME_SCALE = 1.0   # default time scale 
    
    def __init__(self, root: str, n_train: int, n_test: int = 10000, d: int = 200,
                 step: float = 1.0, seed: int = 42, pad_left: int = 1, pad_right: int = 1):
        assert self.SIMULATOR_CLASS is not None, "Must override SIMULATOR_CLASS in child class"
        assert self.SIMULATOR_TYPE is not None, "Must override SIMULATOR_TYPE in child class"
        
        self.root = root
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.d = int(d)
        self.step = float(step)
        self.seed = int(seed)
        self.pad_left = int(pad_left)
        self.pad_right = int(pad_right)
        
        self.data_dir = os.path.join(self.root, self.SIMULATOR_TYPE)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.train_path = os.path.join(self.data_dir, f"train_{self.n_train}_test_{self.n_test}_seed_{self.seed}.csv")
        self.test_path = os.path.join(self.data_dir, f"test_{self.n_train}_test_{self.n_test}_seed_{self.seed}.csv")
        
        # force regeneration for consistency
        self._generate_and_split_save()
        
        self.train_df = self._load_csv(self.train_path)
        self.test_df = self._load_csv(self.test_path)
        
        self.train_data, self.train_duration, self.train_event = self._df_to_arrays(self.train_df)
        self.test_data, self.test_duration, self.test_event = self._df_to_arrays(self.test_df)

        # Convert to days (30.4375 days/month)
        self.train_duration = self.train_duration * self.TIME_SCALE
        self.test_duration = self.test_duration * self.TIME_SCALE
        
        self.n_features = self.d
        self.n_events = int(max(
            len(np.unique(self.train_event)) - 1,
            len(np.unique(self.test_event)) - 1,
        ))
        
        self.train_label = self._duration_to_label(self.train_duration)
        self.test_label = self._duration_to_label(self.test_duration)
        
        max_label = max(self.train_label.max(), self.test_label.max())
        self.n_classes = int(max_label + self.pad_left + self.pad_right)
        self._build_oracle_projection()
    
    def _feature_cols(self):
        return [f"x{i}" for i in range(self.d)]
    
    def _required_cols(self):
        return self._feature_cols() + ["duration", "event"]
    
    def _validate_df(self, df: pd.DataFrame, context: str):
        missing = set(self._required_cols()) - set(df.columns)
        if missing:
            raise ValueError(f"[{context}] Missing columns: {sorted(missing)}")
    
    def _generate_and_split_save(self):
        """Generate full dataset with same seed/beta, then split into train/test."""
        n_total = self.n_train + self.n_test
        
        simulator = self.SIMULATOR_CLASS(n=n_total, d=self.d, seed=self.seed)
        df_full, self.oracle = simulator.generate(**self.DEFAULT_PARAMS)
        
        
        self._validate_df(df_full, context="generated-full")
        
        df_full = df_full.copy()
        for c in self._feature_cols():
            df_full[c] = df_full[c].astype(np.float32)
        df_full["duration"] = df_full["duration"].astype(np.float32)
        df_full["event"] = df_full["event"].astype(np.int64)
        
        train_df = df_full.iloc[:self.n_train].copy()
        test_df = df_full.iloc[self.n_train:].copy()
        
        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load and validate cached CSV."""
        df = pd.read_csv(path)
        self._validate_df(df, context=f"load:{os.path.basename(path)}")
        
        df = df.copy()
        for c in self._feature_cols():
            df[c] = df[c].astype(np.float32)
        df["duration"] = df["duration"].astype(np.float32)
        df["event"] = df["event"].astype(np.int64)
        return df
    
    def _df_to_arrays(self, df: pd.DataFrame):
        """Extract numpy arrays from DataFrame."""
        X = df[self._feature_cols()].values.astype(np.float32)
        duration = df["duration"].values.astype(np.float32)
        event = df["event"].values.astype(np.int64)
        return X, duration, event
    
    def _duration_to_label(self, duration):
        """Discretize continuous durations into bins."""
        bin_idx = (duration // self.step).astype(np.int64) + self.pad_left
        return bin_idx
    
    def get_official_train_test(self):
        """Return train and test PyTorch datasets."""
        train_dataset = SurvivalPyTorchDataset(
            data=self.train_data,
            duration=self.train_duration,
            event=self.train_event,
            label=self.train_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
            duration_to_label=self._duration_to_label,
            survival_at_grids=self.oracle_survival_grid,
        )
        
        test_dataset = SurvivalPyTorchDataset(
            data=self.test_data,
            duration=self.test_duration,
            event=self.test_event,
            label=self.test_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
            duration_to_label=self._duration_to_label,
            survival_at_grids=self.oracle_survival_grid,
        )
        
        return train_dataset, test_dataset
    
    def _build_oracle_projection(self):
        """
        Project dataset-level oracle survival (and link, if any)
        onto the dataset discretization grid.
        """

        oracle = self.oracle

        # -----------------------------
        # Dataset evaluation grid
        # -----------------------------
        n_bins = self.n_classes
        t_grid = (np.arange(n_bins) - self.pad_left) * self.step
        t_grid = np.maximum(t_grid, 0.0)

        self.oracle_time_grid = t_grid.astype(np.float32)

        # -----------------------------
        # PH / PO: piecewise baseline
        # -----------------------------
        if oracle["link"] in {"PH", "PO"}:
            breaks = np.asarray(oracle["breaks"], dtype=float)        # (K+1,)
            survival = np.asarray(oracle["survival"], dtype=float)    # (K,)

            # right-continuous step function
            # survival[k] corresponds to interval (breaks[k], breaks[k+1]]
            idx = np.searchsorted(breaks[1:], t_grid, side="right")
            idx = np.clip(idx, 0, len(survival) - 1)

            self.oracle_survival_grid = survival[idx].astype(np.float32)
            return

        # -----------------------------
        # GEN: discrete-time baseline
        # -----------------------------
        if oracle["link"] == "Gen":
            taus = np.asarray(oracle["taus"], dtype=float)             # (m+1,)
            survival = np.asarray(oracle["survival"], dtype=float)     # (m,)

            idx = np.searchsorted(taus[1:], t_grid, side="right")
            idx = np.clip(idx, 0, len(survival) - 1)

            self.oracle_survival_grid = survival[idx].astype(np.float32)

            # optional: link recovery
            if "link_z" in oracle:
                self.oracle_link_z = np.asarray(oracle["link_z"], dtype=np.float32)
                self.oracle_link_values = np.asarray(
                    oracle["link_values"], dtype=np.float32
                )
            return

        raise RuntimeError(f"Unknown oracle type: {oracle.get('type')}")


class PHPiecewiseDataset(BaseSurvivalDataset):
    SIMULATOR_CLASS = ProportionalHazardsPiecewiseSimulator
    SIMULATOR_TYPE = "ph_piecewise"
    DEFAULT_PARAMS = {
        "target_censor": 0.6,
        "censor_admin_frac": 0.5,
        "censor_admin_q": 0.85,
        "n_nonzero": 10,
        "beta_scale": 0.3,
        "x_dist": "normal",
        "corr": 0.5,
        "heavy_tail_df": 4.0,
        "mix_prob": 0.2,
        "mix_shift": 1.5,
        "n_knots": 256,
        "t_max": 400.0,
        "hazard_level": 0.005,
        "hazard_jitter": 0.2,
    }


class POPiecewiseDataset(BaseSurvivalDataset):
    SIMULATOR_CLASS = ProportionalOddsPiecewiseSimulator
    SIMULATOR_TYPE = "po_piecewise"
    DEFAULT_PARAMS = {
        "target_censor": 0.6,
        "censor_admin_frac": 0.5,
        "censor_admin_q": 0.85,
        "n_nonzero": 10,
        "beta_scale": 0.3,
        "x_dist": "normal",
        "corr": 0.5,
        "heavy_tail_df": 4.0,
        "mix_prob": 0.2,
        "mix_shift": 1.5,
        "n_knots": 256,
        "t_max": 400.0,
        "odds_level": 0.01,
        "odds_jitter": 0.2,
    }


class LinkRecoveryHardDataset(BaseSurvivalDataset):
    SIMULATOR_CLASS = DiscreteTimeLinkSimulatorHard
    SIMULATOR_TYPE = "link_recovery_hard"
    TIME_SCALE = 20000.
    DEFAULT_PARAMS = {
        "m": 60,
        "tau_max": 20.0,
        "target_censor": 0.6,
        "censor_admin_frac": 0.5,
        "censor_admin_q": 0.85,
        "n_nonzero": 10,
        "beta_scale": 0.35,
        "x_dist": "normal",
        "corr": 0.5,
        "heavy_tail_df": 4.0,
        "mix_prob": 0.2,
        "mix_shift": 1.5,
        "z_min": -8.0,
        "z_max": 8.0,
        "L_star": 24,
        "eta_warp": 2.0,
        "eta_bumps": 3,
    }


if __name__ == "__main__":
    # Test PH dataset
    ph_data = PHPiecewiseDataset(root="./sim_data", n_train=5000, n_test=10000, step=30.0, seed=42)
    ph_train, ph_test = ph_data.get_official_train_test()
    
    print(f"PH Training: {len(ph_train)} samples, {ph_train.n_features} features, {ph_train.n_classes} classes")
    print(f"PH Test: {len(ph_test)} samples")
    print(f"Sample keys: {ph_train[0].keys()}")
    
    # Test PO dataset
    po_data = POPiecewiseDataset(root="./sim_data", n_train=5000, n_test=10000, step=30.0, seed=42)
    po_train, po_test = po_data.get_official_train_test()
    
    print(f"\nPO Training: {len(po_train)} samples")
    
    # Test Link Recovery dataset
    link_data = LinkRecoveryHardDataset(root="./sim_data", n_train=5000, n_test=10000, step=30.0, seed=42)
    link_train, link_test = link_data.get_official_train_test()
    
    print(f"\nLink Recovery Training: {len(link_train)} samples")