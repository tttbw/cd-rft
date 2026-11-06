"""Measure the control matrix C of a checkpoint on several probe seeds.

  python measure/control_matrix.py --ckpt <checkpoint> --name <tag>
writes <out>/C_<name>_s<seed>_L<trunc>.npz with C [M, K], per-family log-likelihoods and family names.
--trunc L keeps the first L target tokens of every probe sample (length-matched probes; 0 = full targets).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "method"))
from control_reg import control_matrix  # noqa: E402
from data import build_bshared_probe     # noqa: E402
from geometry import geometry            # noqa: E402


def truncate(item, L):
    if L <= 0:
        return item
    m = item["target_mask"]
    idx = [i for i, v in enumerate(m) if v]
    if len(idx) <= L:
        return item
    cut = idx[L - 1] + 1
    return {**item, "input_ids": item["input_ids"][:cut], "target_mask": m[:cut]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/control")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--trunc", type=int, nargs="+", default=[0])
    ap.add_argument("--npf", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=448)
    ap.add_argument("--gate_level", choices=["sublayer", "layer"], default="sublayer")
    ap.add_argument("--tok", default=None, help="tokenizer source (default: the checkpoint)")
    ap.add_argument("--prompt_version", default="measurement", help="'none' = training system prompts")
    a = ap.parse_args()
    pv = None if a.prompt_version.lower() == "none" else a.prompt_version

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok or a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    os.makedirs(a.out, exist_ok=True)
    for s in range(a.seeds):
        probe = build_bshared_probe(tok, n_per_family=a.npf, seed=s, max_len=a.max_len,
                                    families=("math", "code", "logic"), prompt_version=pv)
        for L in a.trunc:
            C, ell, fams = control_matrix(model, [truncate(r, L) for r in probe], a.gate_level)
            g = geometry(C)
            np.savez(os.path.join(a.out, f"C_{a.name}_s{s}_L{L}.npz"), C=C, ell=np.asarray(ell),
                     fams=np.array(fams), B_shared=g["B_shared"], seed=s, trunc=L, npf=a.npf)
            print(f"[{a.name}] seed={s} L={L} " + " ".join(f"{k}={v:.4f}" for k, v in g.items()
                                                           if not isinstance(v, (list, dict))), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
