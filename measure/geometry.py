"""Geometry of a control matrix C [M, K] (rows = task families, columns = gated sublayers).

  B_shared = lambda_max(C C^T) / tr(C C^T)          spectral concentration
  B_dir    = lambda_max(R) / M                      R = cosine matrix of the rows (direction only)
  B_norm   = max_m ||C_m||^2 / sum_m ||C_m||^2       row-norm term
  PR_m     = (sum_k |C_mk|)^2 / sum_k C_mk^2         participation ratio of row m, in [1, K]

  python measure/geometry.py results/control/C_*_L0.npz
"""
import sys

import numpy as np


def geometry(C, fams=None):
    C = np.asarray(C, dtype=np.float64)
    M = C.shape[0]
    G = C @ C.T
    d = np.diag(G).copy()
    out = {"B_shared": float(np.linalg.eigvalsh(G).max() / G.trace())}
    nz = d > 0
    Cn = C[nz] / np.sqrt(d[nz])[:, None]
    R = Cn @ Cn.T
    out["B_dir"] = float(np.linalg.eigvalsh(R).max() / R.shape[0])
    out["B_norm"] = float((d / d.sum()).max())
    out["PR"] = [float(np.abs(C[m]).sum() ** 2 / ((C[m] ** 2).sum() + 1e-12)) for m in range(M)]
    if fams is not None and nz.all():
        out["abs_cos"] = {f"{fams[i]}-{fams[j]}": abs(float(R[i, j]))
                          for i in range(M) for j in range(i + 1, M)}
    return out


def random_reference(M=3, K=56, draws=50000, seed=0):
    """B_dir and mean |cos| of i.i.d. Gaussian rows of the same shape."""
    rng = np.random.default_rng(seed)
    bd, cs = np.empty(draws), np.empty(draws)
    for t in range(draws):
        X = rng.standard_normal((M, K))
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        R = Xn @ Xn.T
        bd[t] = np.linalg.eigvalsh(R)[-1] / M
        cs[t] = np.abs(R[np.triu_indices(M, 1)]).mean()
    return {"B_dir": float(bd.mean()), "mean_abs_cos": float(cs.mean())}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        z = np.load(path, allow_pickle=True)
        print(path, geometry(z["C"], [str(f) for f in z["fams"]]))
