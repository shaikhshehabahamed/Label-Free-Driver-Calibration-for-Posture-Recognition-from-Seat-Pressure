"""CUDA learners for the calibration-ladder study (GTX 1650, 4 GB: everything here uses tiny kernels / small batches).

GpuLDA   StandardScaler -> PCA(k) -> LDA(solver='lsqr', shrinkage='auto' | float), re-implemented in torch with scikit-learn 1.9's
         formulas (covariance-eigh PCA, per-class standardisation + Ledoit-Wolf).  Checked against scikit-learn in 30_validate_gpu.py.
         prepare() does the expensive eigendecomposition once, fit_k() is then cheap -> nested tuning over (k, shrinkage) is nearly free.
xgb_predict   gradient-boosted trees (XGBoost 3.x, device='cuda', histogram method) - the non-linear learner (replaces CPU random forest).
cnn_predict   3-layer CNN on the pooled map (torch, CUDA).
loso_eval     leave-one-driver-out loop with a thermal guard between folds.
"""
import guard  # noqa: F401  (thread caps, priority) - must come before numpy / torch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=DEV)


# ------------------------------------------------------------------------------------------------ linear discriminant
def _lw_cov(X):
    """Ledoit-Wolf shrunk covariance of the (re-centred) data, X (n, p) float64 - sklearn.covariance.ledoit_wolf."""
    n, p = X.shape
    Xc = X - X.mean(0, keepdim=True)
    emp = Xc.T @ Xc / n
    X2 = Xc ** 2
    emp_trace = X2.sum(0) / n
    mu = emp_trace.sum() / p
    delta_ = (emp ** 2).sum()
    beta_ = (X2.sum(1) ** 2).sum()
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * emp_trace.sum() + p * mu ** 2) / p
    beta = torch.minimum(beta, delta)
    shrink = torch.zeros((), dtype=X.dtype, device=X.device) if float(beta) == 0.0 else beta / delta
    return (1 - shrink) * emp + shrink * mu * torch.eye(p, dtype=X.dtype, device=X.device)


def _class_cov(Xg, shrinkage):
    """sklearn.discriminant_analysis._cov for one class."""
    if isinstance(shrinkage, str):                                    # 'auto'
        mean = Xg.mean(0, keepdim=True)
        sd = Xg.std(0, unbiased=False)
        sd = torch.where(sd == 0, torch.ones_like(sd), sd)
        s = _lw_cov((Xg - mean) / sd)
        return sd[:, None] * s * sd[None, :]
    Xc = Xg - Xg.mean(0, keepdim=True)
    emp = Xc.T @ Xc / Xg.shape[0]
    p = emp.shape[0]
    mu = torch.trace(emp) / p
    return (1 - shrinkage) * emp + shrinkage * mu * torch.eye(p, dtype=emp.dtype, device=emp.device)


class GpuLDA:
    def prepare(self, Xtr):
        X = _t(Xtr)
        self.mu = X.mean(0)
        sd = X.std(0, unbiased=False)
        self.sd = torch.where(sd == 0, torch.ones_like(sd), sd)
        self.Xs = (X - self.mu) / self.sd
        C = (self.Xs.T @ self.Xs).double() / (self.Xs.shape[0] - 1)
        w, V = torch.linalg.eigh(C)
        self.V = V.flip(1).float()                                      # descending eigenvalues
        self.w = w.flip(0)
        return self

    def fit_k(self, ytr, k, shrinkage="auto", rank_tol=None):
        """rank_tol (opt-in; the pilot's results use None): drop PCA directions whose eigenvalue is below rank_tol x the largest one - the numerical-rank
        tolerance of scikit-learn's LDA (`tol`).  Needed when the features have exactly collinear / constant directions (e.g. whitened or per-cell scaled maps)."""
        k = min(k, self.V.shape[1])
        if rank_tol is not None:
            k = max(1, min(k, int((self.w > rank_tol * self.w[0]).sum())))
        self.k = k
        Z = (self.Xs @ self.V[:, :k]).double()
        self.classes = np.unique(ytr)
        yt = torch.as_tensor(np.searchsorted(self.classes, ytr), device=DEV)
        n = Z.shape[0]
        nc = len(self.classes)
        means = torch.stack([Z[yt == c].mean(0) for c in range(nc)])
        priors = torch.bincount(yt, minlength=nc).double() / n
        cov = torch.zeros(k, k, dtype=torch.float64, device=DEV)
        for c in range(nc):
            cov += priors[c] * _class_cov(Z[yt == c], shrinkage)
        coef = torch.linalg.solve(cov, means.T).T
        self.coef = coef
        self.intercept = -0.5 * torch.diagonal(means @ coef.T) + torch.log(priors)
        return self

    def predict(self, Xte):
        Z = ((_t(Xte) - self.mu) / self.sd) @ self.V[:, :self.k]
        s = Z.double() @ self.coef.T + self.intercept
        return self.classes[s.argmax(1).cpu().numpy()]


# ------------------------------------------------------------------------------------------------ gradient-boosted trees
def xgb_predict(Xtr, ytr, Xte, n_estimators=80, max_depth=4, lr=0.2, colsample=0.3, seed=0):
    """fixed, modest-capacity boosted trees (NOT tuned: it is a non-linear robustness check, kept small to spare the laptop)."""
    import xgboost as xgb
    classes = np.unique(ytr)
    m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, subsample=0.8,
                          colsample_bytree=colsample, min_child_weight=5, reg_lambda=2.0, tree_method="hist",
                          device="cuda", max_bin=32, n_jobs=2, random_state=seed, verbosity=0)
    m.fit(np.asarray(Xtr, np.float32), np.searchsorted(classes, ytr))
    return classes[m.predict(np.asarray(Xte, np.float32))]


# ------------------------------------------------------------------------------------------------ small CNN
class Net(nn.Module):
    def __init__(self, n_cls=20, in_ch=1):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(c(in_ch, 16), c(16, 32), nn.MaxPool2d(2), c(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3))
        self.h = nn.Linear(64, n_cls)

    def forward(self, x):
        return self.h(self.f(x))


def cnn_predict(Xtr, ytr, Xte, shape=(16, 40), epochs=14, seed=0, batch=256, in_ch=1):
    """in_ch > 1: the columns of X are the channels stacked (channel-major); each channel is standardised separately (for in_ch = 1 this is the original global standardisation)"""
    torch.manual_seed(seed)
    Xtr3 = np.asarray(Xtr, np.float32).reshape(-1, in_ch, *shape)
    Xte3 = np.asarray(Xte, np.float32).reshape(-1, in_ch, *shape)
    mu = Xtr3.mean(axis=(0, 2, 3), keepdims=True)
    sd = Xtr3.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    xt = _t((Xtr3 - mu) / sd)
    yt = torch.as_tensor(ytr - 1, dtype=torch.long, device=DEV)
    xe = _t((Xte3 - mu) / sd)
    net = Net(in_ch=in_ch).to(DEV)
    steps = epochs * ((len(xt) + batch - 1) // batch)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.cross_entropy(net(xt[b]), yt[b]).backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([net(xe[i:i + 2048]).argmax(1) for i in range(0, len(xe), 2048)]).cpu().numpy() + 1


def cnn_predict_adabn(Xtr, ytr, Xte, shape=(16, 40), epochs=14, seed=0, batch=256, in_ch=1):
    """Same network, same training loop and seed handling as `cnn_predict` (kept separate so the pilot's function is untouched); returns
    (standard predictions, AdaBN predictions).  AdaBN (Li et al. 2018, Algorithm 1): the batch-normalisation statistics of every BN layer are re-estimated
    on the TARGET domain's own unlabelled inputs (here: all rows of the held-out driver in one batch = exact mean / variance) and used instead of the
    source statistics; weights are untouched and no labels are used."""
    torch.manual_seed(seed)
    Xtr3 = np.asarray(Xtr, np.float32).reshape(-1, in_ch, *shape)
    Xte3 = np.asarray(Xte, np.float32).reshape(-1, in_ch, *shape)
    mu = Xtr3.mean(axis=(0, 2, 3), keepdims=True)
    sd = Xtr3.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    xt = _t((Xtr3 - mu) / sd)
    yt = torch.as_tensor(ytr - 1, dtype=torch.long, device=DEV)
    xe = _t((Xte3 - mu) / sd)
    net = Net(in_ch=in_ch).to(DEV)
    steps = epochs * ((len(xt) + batch - 1) // batch)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.cross_entropy(net(xt[b]), yt[b]).backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        std = torch.cat([net(xe[i:i + 2048]).argmax(1) for i in range(0, len(xe), 2048)]).cpu().numpy() + 1
    import copy
    ada = copy.deepcopy(net)
    for m in ada.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                                   # cumulative average: the single forward pass below sets mean / variance exactly
    ada.train()
    for m in ada.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    with torch.no_grad():
        ada(xe)                                                  # one batch = the whole held-out driver (labels never used)
    ada.eval()
    with torch.no_grad():
        adabn = torch.cat([ada(xe[i:i + 2048]).argmax(1) for i in range(0, len(xe), 2048)]).cpu().numpy() + 1
    return std, adabn


# ------------------------------------------------------------------------------------------------ evaluation loop
def predict_fold(model, Xtr, ytr, Xte, kw=None):
    kw = kw or {}
    if model == "gLDA":
        m = GpuLDA().prepare(Xtr).fit_k(ytr, kw.get("pca", 150), kw.get("shrinkage", "auto"), kw.get("rank_tol"))
        return m.predict(Xte)
    if model == "XGB":
        return xgb_predict(Xtr, ytr, Xte, **kw)
    if model == "CNN":
        return cnn_predict(Xtr, ytr, Xte, **kw)
    raise ValueError(model)


def loso_eval(model, X, y, u, kw=None, tag="", keep_pred=False):
    """leave-one-driver-out; rows with NaN features are dropped.  Returns the per-driver DataFrame
    (and, with keep_pred=True, also the retained (y, u, predicted y) arrays)."""
    ok = ~np.isnan(X).any(axis=1)
    X, y, u = X[ok], y[ok], u[ok]
    out = []
    pred_all = np.zeros(len(y), int)
    for s in np.unique(u):
        te = u == s
        pred = predict_fold(model, X[~te], y[~te], X[te], kw)
        pred_all[te] = pred
        out.append(dict(u=int(s), acc=accuracy_score(y[te], pred), bal_acc=balanced_accuracy_score(y[te], pred),
                        f1=f1_score(y[te], pred, average="macro", zero_division=0), n=int(te.sum())))
        guard.cooldown(tag=f"{model} {tag} driver {s}")
    d = pd.DataFrame(out)
    return (d, y, u, pred_all) if keep_pred else d
