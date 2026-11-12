"""Unit tests of the adaptation operators in calib_lab.py against authoritative implementations.

 (a) whitening algebra      M^{-1/2} x   vs  pyRiemann `invsqrtm` (official library of the Riemannian-alignment / Euclidean-alignment line of work)
 (b) Euclidean alignment    aligned trial covariances M^{-1/2} R_i M^{-1/2}  vs  pyRiemann `TLCenter(metric='euclid')`
 (c) CORAL                  shrinkage 0.5 on z-scored features == (cov + I) up to a constant:  whitening with a = 0.5  vs  CORAL's official  cov(X) + eye  (Ex_CORAL.m)
 (d) window statistics      GPU causal mean / covariance of the last w frames (expanding at the start)  vs  NumPy
 (e) cmvn                   per-cell scaling vs a direct NumPy computation
pyRiemann was installed to a scratch folder (`pip install --no-deps --target ...`), the shared venv is untouched.
usage: python 55_verify_operators.py <path to the folder holding pyriemann>
"""
import sys

import guard  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import warnings

warnings.filterwarnings("ignore")
import calib_lab as L
from pyriemann.geometry.base import invsqrtm
from pyriemann.transfer import TLCenter, encode_domains

rng = np.random.default_rng(0)
d, R = 24, 6
res = {}

# (a) whitening algebra
A = rng.normal(size=(d, 400))
M = A @ A.T / 400 + 0.1 * np.eye(d)
T = rng.normal(size=(R, d))
mine = L._inv_sqrt_apply(torch.tensor(M)[None].repeat(R, 1, 1), torch.tensor(T), alpha=0.0).numpy()
ref = (invsqrtm(M) @ T.T).T
res["a_whitening_vs_pyriemann_invsqrtm"] = float(np.abs(mine - ref).max())

# (b) EA == TLCenter(euclid) on trial covariances: trials = consecutive blocks of 41 frames of a mixed-posture stream
Xs = rng.normal(size=(41 * 40, d)) @ rng.normal(size=(d, d)) * 0.3 + rng.normal(size=d)
trials = Xs.reshape(40, 41, d)
covs = np.einsum("ntd,nte->nde", trials, trials) / 41                         # trial second-moment matrices (EA: R_i = X_i X_i^T)
Mbar = covs.mean(0)
Msq = invsqrtm(Mbar)
ea_cov = np.einsum("de,nef,fg->ndg", Msq, covs, Msq)                           # covariance of the aligned trials
y_enc = encode_domains(covs, np.zeros(40, int), np.array(["tgt"] * 40))[1]
tl = TLCenter(target_domain="tgt", metric="euclid").fit(covs, y_enc)
ref_cov = tl.transform(covs)
res["b_EA_vs_TLCenter_euclid_covariances"] = float(np.abs(ea_cov - ref_cov).max())
# and our operator on the data itself: whitened frames have mean second-moment = I (alpha = 0, all frames of the window in one block)
Xt = torch.tensor(Xs, dtype=torch.float64)
Cw = (Xt.T @ Xt / Xt.shape[0])
Xw = L._inv_sqrt_apply(Cw[None].repeat(len(Xt), 1, 1), Xt, 0.0).double().numpy()
res["b_EA_aligned_second_moment_minus_I"] = float(np.abs(Xw.T @ Xw / len(Xw) - np.eye(d)).max())   # all frames: the aligned data have identity second moment

# (c) CORAL: z-score, cov + I (official Ex_CORAL.m), vs shrinkage a = 0.5 whitening (tr(C)/d = 1 for z-scored data -> (C + I)/2)
Xz = (Xs - Xs.mean(0)) / Xs.std(0)
C = np.cov(Xz, rowvar=False)                                                    # MATLAB cov is the unbiased estimate
coral = (invsqrtm(C + np.eye(d)) @ (Xz[:R] - 0).T).T                            # source whitening half of CORAL: Xs * (cov(Xs)+I)^(-1/2)
mine_c = L._inv_sqrt_apply(torch.tensor(C)[None].repeat(R, 1, 1), torch.tensor(Xz[:R]), alpha=0.5).numpy() / np.sqrt(2.0)
res["c_CORAL_official_vs_shrink0.5_over_sqrt2 (unbiased cov: tr/d = n/(n-1))"] = float(np.abs(coral - mine_c).max())
Cb = np.cov(Xz, rowvar=False, bias=True)                                        # population covariance: tr(C)/d = 1 exactly for z-scored data
coral_b = (invsqrtm(Cb + np.eye(d)) @ Xz[:R].T).T
mine_b = L._inv_sqrt_apply(torch.tensor(Cb)[None].repeat(R, 1, 1), torch.tensor(Xz[:R]), alpha=0.5).numpy() / np.sqrt(2.0)
res["c_CORAL_vs_shrink0.5_over_sqrt2 (population cov, exact algebra)"] = float(np.abs(coral_b - mine_b).max())

# (d) window statistics vs NumPy (expanding at the start)
n, w = 2000, 300
S = rng.normal(size=(n, d)).astype(np.float32) + np.linspace(0, 1, n)[:, None].astype(np.float32)
pos = np.array([0, 1, 50, 299, 300, 301, 1200, 1999])
m_g, C_g = L.window_stats(torch.tensor(S, device=L.DEV), torch.tensor(pos, device=L.DEV), w)
err_m = err_c = 0.0
for k, p in enumerate(pos):
    W_ = S[max(0, p - w + 1):p + 1].astype(np.float64)
    err_m = max(err_m, np.abs(m_g[k].cpu().numpy() - W_.mean(0)).max())
    err_c = max(err_c, np.abs(C_g[k].cpu().numpy() - np.cov(W_, rowvar=False, bias=True)).max() if len(W_) > 1 else 0.0)
res["d_window_mean_max_abs_err"] = float(err_m)
res["d_window_cov_max_abs_err"] = float(err_c)
cm = L.causal_mean_t(torch.tensor(S, device=L.DEV), 41).cpu().numpy()
ref_cm = L.roll_mean(S, 41)
res["d_causal_mean_gpu_vs_pandas"] = float(np.abs(cm - ref_cm).max())

# (e) cmvn
p0 = torch.tensor([1500], device=L.DEV)
Zr = torch.tensor(S[1500:1501], device=L.DEV)
m1, C1 = L.window_stats(torch.tensor(S, device=L.DEV), p0, w)
out = L.apply_operator("cmvn", Zr, m1, C1).cpu().numpy()[0]
Wn = S[1500 - w + 1:1501].astype(np.float64)
v = Wn.var(0)
ref_out = (S[1500] - Wn.mean(0)) / np.sqrt(v + (0.1 * np.sqrt(v).mean()) ** 2)
res["e_cmvn_vs_numpy"] = float(np.abs(out - ref_out).max())

for k, v in res.items():
    flag = "OK " if v < 5e-4 else "CHECK"
    print(f"{flag} {k:48s} {v:.3e}")
