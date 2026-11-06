"""Control coefficients over gated sublayers and the first-order CD-RFT proxy.

Gates g_k (nominal value 1) scale the output of every attention and MLP sublayer. For task family m
with mean reference-target log-likelihood J_m, the control coefficient is C_{m,k} = (dJ_m/dg_k) / J_m.
The shared control bottleneck is B_shared = lambda_max(C C^T) / tr(C C^T).
"""
import torch
import torch.nn.functional as F


def _decoder_layers(model):
    m = model
    for attr in ("model", "module"):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        m = getattr(m, attr, m)
    for path in (("model", "layers"), ("language_model", "model", "layers"),
                 ("model", "language_model", "layers"), ("language_model", "layers")):
        obj, ok = model, True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            return obj
    raise AttributeError("cannot locate decoder layers")


def _gate_targets(layer, level):
    if level == "layer":
        return [("blk", layer)]
    out = []
    if hasattr(layer, "self_attn"):
        out.append(("attn", layer.self_attn))
    if hasattr(layer, "mlp"):
        out.append(("mlp", layer.mlp))
    return out


def _scale_hook(s):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] * s,) + tuple(output[1:])
        return output * s
    return hook


class BlockGates:
    """Differentiable scalar gates (value 1) on every sublayer output."""

    def __init__(self, model, level, device, dtype):
        self.handles, self.gates, self.names = [], [], []
        for i, layer in enumerate(_decoder_layers(model)):
            for tag, sub in _gate_targets(layer, level):
                g = torch.ones((), device=device, dtype=dtype, requires_grad=True)
                self.gates.append(g)
                self.names.append(f"L{i}.{tag}")
                self.handles.append(sub.register_forward_hook(_scale_hook(g)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


class _GateConst:
    """Constant per-gate scales (no gradient with respect to the gates)."""

    def __init__(self, model, scales, level):
        self.handles, k = [], 0
        for layer in _decoder_layers(model):
            for _, sub in _gate_targets(layer, level):
                s = scales[k] if k < len(scales) else None
                k += 1
                if s is not None:
                    self.handles.append(sub.register_forward_hook(_scale_hook(s)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class _OutProbe:
    """Keeps the graph-connected output of every gated sublayer."""

    def __init__(self, model, level):
        self.handles, self.acts = [], []
        for layer in _decoder_layers(model):
            for _, sub in _gate_targets(layer, level):
                k = len(self.acts)
                self.acts.append(None)
                self.handles.append(sub.register_forward_hook(self._mk(k)))

    def _mk(self, k):
        def h(mod, inp, out):
            self.acts[k] = out[0] if isinstance(out, tuple) else out
        return h

    def remove(self):
        for h in self.handles:
            h.remove()


def _seq_logps(model, input_ids, attention_mask, target_mask):
    """Sum of token log-probabilities over the target positions, per sequence."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tok_lp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = target_mask[:, 1:].to(tok_lp.dtype)
    return (tok_lp * mask).sum(dim=-1)


def _seq_logps_gated(model, ids, attn, tmask, scales, level):
    if scales is None:
        return _seq_logps(model, ids, attn, tmask).mean()
    gc = _GateConst(model, scales, level=level)
    try:
        return _seq_logps(model, ids, attn, tmask).mean()
    finally:
        gc.remove()


def family_batch(probe, fam, device):
    rows = [p for p in probe if p["task"] == fam]
    L = max(len(r["input_ids"]) for r in rows)
    ids = torch.zeros((len(rows), L), dtype=torch.long, device=device)
    attn = torch.zeros((len(rows), L), dtype=torch.long, device=device)
    tmask = torch.zeros((len(rows), L), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"], device=device)
        attn[i, :n] = 1
        tmask[i, :n] = torch.tensor(r["target_mask"], device=device)
    return ids, attn, tmask


def control_matrix(model, probe, level):
    """Control matrix C [M, K] on a fixed probe (rows ordered by sorted family name)."""
    device = next(model.parameters()).device
    fams = sorted(set(p["task"] for p in probe))
    gates = BlockGates(model, level=level, device=device, dtype=next(model.parameters()).dtype)
    rows, ells = [], []
    try:
        for fam in fams:
            ids, attn, tmask = family_batch(probe, fam, device)
            J = _seq_logps(model, ids, attn, tmask).mean()
            g = torch.autograd.grad(J, gates.gates, allow_unused=True)
            gv = torch.stack([(x if x is not None else torch.zeros((), device=device, dtype=J.dtype))
                              for x in g])
            rows.append((gv / J).detach().float().cpu())
            ells.append(float(J.detach()))
    finally:
        gates.remove()
    return torch.stack(rows).double().numpy(), ells, fams


_PLACEBO_GEN = None


def _placebo_W(Wmat, mode, seed):
    """Replace each row W_m by an equal-norm direction orthogonal to it (mode: perm | rand)."""
    global _PLACEBO_GEN
    if _PLACEBO_GEN is None:
        _PLACEBO_GEN = torch.Generator(device="cpu")
        _PLACEBO_GEN.manual_seed(int(seed) + 1000003)
    W = Wmat.detach().float().cpu()
    M, K = W.shape
    out = W.clone()
    post = []
    for m in range(M):
        w = W[m]
        wn = w.norm()
        if float(wn) < 1e-12:
            continue
        for _ in range(8):
            v = w[torch.randperm(K, generator=_PLACEBO_GEN)] if mode == "perm" \
                else torch.randn(K, generator=_PLACEBO_GEN)
            v = v - (v @ w) / (wn * wn) * w
            vn = v.norm()
            if float(vn) > 1e-6 * float(wn):
                break
        v = v * (wn / vn)
        out[m] = v
        post.append(float((v @ w).abs() / (v.norm() * wn + 1e-12)))
    stats = {"placebo": 1.0 if mode == "perm" else 2.0,
             "placebo_cos": sum(post) / max(1, len(post))}
    return out.to(device=Wmat.device, dtype=Wmat.dtype), stats


def bshared_proxy_penalty(model, probe, lambda2, fd_eps, fd_samples, denom_shift, level,
                          placebo=None, seed=0):
    """First-order proxy for B_shared on the probe; returns (penalty, stats).

    penalty = lambda2 * mean_m <W_m, C_m(theta)>, with W = dR/dC evaluated at the detached control
    matrix and R = tr(G^2) / tr(G)^2, G = C C^T. <W_m, C_m> is computed by a central difference in
    gate space, so a single ordinary backward pass gives its parameter gradient.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    fams = sorted(set(p["task"] for p in probe))
    gc_was = bool(getattr(model, "is_gradient_checkpointing", False))
    if gc_was and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    try:
        return _proxy_core(model, probe, fams, lambda2, fd_eps, fd_samples, denom_shift, level,
                           placebo, seed, device, dtype)
    finally:
        if gc_was and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


def _proxy_core(model, probe, fams, lambda2, eps_fd, pbs, eps, level, placebo, seed, device, dtype):
    cbar_rows, kept = [], []
    for fam in fams:
        ids, attn, tmask = family_batch(probe, fam, device)
        ap = _OutProbe(model, level)
        try:
            J = _seq_logps(model, ids, attn, tmask).mean()
            if not torch.isfinite(J) or J.abs() < eps:
                continue
            outs = [a for a in ap.acts if a is not None]
            gouts = torch.autograd.grad(J, outs, retain_graph=False, allow_unused=True,
                                        create_graph=False)
            cbar = []
            for o, g in zip(outs, gouts):
                cbar.append(0.0 if g is None else float((g.detach() * o.detach()).sum() / J.detach()))
            cbar_rows.append(torch.tensor(cbar, device=device))
            kept.append(fam)
        finally:
            ap.remove()
    if len(cbar_rows) < 2:
        return torch.zeros((), device=device), {"skipped": 1.0}
    K = min(len(r) for r in cbar_rows)
    Cbar = torch.stack([r[:K] for r in cbar_rows]).float()
    G = Cbar @ Cbar.t()
    tr = G.trace()
    evals = torch.linalg.eigvalsh(G)
    B_shared_val = float((evals[-1] / (tr + 1e-12)).detach())

    trm = torch.diagonal(G).sum() + 1e-12
    R = (G @ G).diagonal().sum() / (trm * trm)
    Wmat = ((4.0 / (trm * trm)) * (G @ Cbar - R * trm * Cbar)).detach()
    plc_stats = {}
    if placebo in ("perm", "rand"):
        Wmat, plc_stats = _placebo_W(Wmat, placebo, seed)

    penalty = torch.zeros((), device=device, dtype=dtype)
    for mi, fam in enumerate(kept):
        ids, attn, tmask = family_batch(probe, fam, device)
        ids, attn, tmask = ids[:pbs], attn[:pbs], tmask[:pbs]
        w_m = Wmat[mi]
        wn = w_m.norm()
        if float(wn) < 1e-12:
            continue
        wh = w_m / wn
        Jp = _seq_logps_gated(model, ids, attn, tmask, 1.0 + eps_fd * wh, level)
        Jn = _seq_logps_gated(model, ids, attn, tmask, 1.0 - eps_fd * wh, level)
        J0 = _seq_logps_gated(model, ids, attn, tmask, None, level)
        Dw = wn * (Jp - Jn) / (2.0 * eps_fd)
        J0s = J0 + (eps if float(J0) >= 0 else -eps)
        penalty = penalty + (Dw / J0s) / len(kept)
    penalty = lambda2 * penalty
    return penalty, {"B_shared": B_shared_val, "moment_R": float(R.detach()),
                     "n_fam": float(len(kept)), **plc_stats}
