"""CD-RFT trainer: GRPO followed by a projection of B_shared on a fixed probe after each optimizer step."""
import json

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer

from control_reg import bshared_proxy_penalty


class _ProjectionCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self._trainer._run_projection()


class CDRFTTrainer(GRPOTrainer):
    """cdrft_cfg: the CD-RFT keys of the run config. placebo: None | perm | rand.
    dose_schedule: optional JSON {step: n} replaying a fixed number of projection updates per step."""

    def __init__(self, *args, cdrft_cfg, bshared_probes, seed, placebo=None, dose_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cdrft_cfg
        self.bshared_probes = list(bshared_probes)
        self.placebo, self.seed = placebo, seed
        self._warned = False
        self._rot = 0
        self._schedule = None
        if dose_schedule:
            with open(dose_schedule) as f:
                self._schedule = {int(k): int(v) for k, v in json.load(f).items()}
        if self.bshared_probes and self.cfg["PROXY_MODE"] == "indep":
            self.add_callback(_ProjectionCallback(self))

    def _proxy(self, probe):
        c = self.cfg
        return bshared_proxy_penalty(self.model, probe, lambda2=c["lambda2"], fd_eps=c["PROXY_FD_EPS"],
                                     fd_samples=c["PROXY_FD_SAMPLES"], denom_shift=c["PROXY_DENOM_SHIFT"],
                                     level=c["gate_level"], placebo=self.placebo, seed=self.seed)

    def _run_projection(self):
        """While B_shared(probe) > PROXY_TARGET, take normalized proxy steps (at most PROXY_KMAX per step)."""
        if not self.bshared_probes:
            return
        model = self.model
        was_train = model.training
        model.train()
        target = float(self.cfg["PROXY_TARGET"])
        kmax = int(self.cfg["PROXY_KMAX"])
        lr = float(self.cfg["PROXY_LR"])
        params = [p for p in model.parameters() if p.requires_grad]
        probes = self.bshared_probes
        try:
            if len(probes) > 1:
                vals = []
                for pr in probes:
                    pen, st = self._proxy(pr)
                    vals.append(float(st.get("B_shared", 0.0)))
                    del pen
                avg = sum(vals) / len(vals)
                if avg == avg and avg <= target:
                    if self.accelerator.is_main_process:
                        self.log({"cdrft/B_shared": avg, "cdrft/proxy_inner_iters": 0.0})
                    return
            used, last = 0, None
            n_sched = None if self._schedule is None else \
                self._schedule.get(int(self.state.global_step), 0)
            for it in range(kmax if n_sched is None else n_sched + 1):
                pen, stats = self._proxy(probes[(self._rot + it) % len(probes)])
                last = stats
                b = float(stats.get("B_shared", 0.0))
                if (n_sched is None and b <= target and b == b) or (n_sched is not None and it >= n_sched):
                    break
                if b != b or not torch.is_tensor(pen) or not pen.requires_grad:
                    break
                g = torch.autograd.grad(pen, params, allow_unused=True, retain_graph=False)
                norm = torch.sqrt(sum((gg.float() ** 2).sum() for gg in g if gg is not None)) + 1e-12
                if not torch.isfinite(norm):
                    break
                with torch.no_grad():
                    scale = (lr / norm).item()
                    for p, gg in zip(params, g):
                        if gg is not None:
                            p.add_(gg.to(p.dtype), alpha=-scale)
                used += 1
            self._rot = (self._rot + 1) % len(probes)
            if last is not None and self.accelerator.is_main_process:
                self.log({f"cdrft/{k}": v for k, v in last.items()})
                self.log({"cdrft/proxy_inner_iters": float(used)})
        except Exception as e:
            if not self._warned:
                print(f"[CDRFT] projection step skipped ({type(e).__name__}: {e})")
                self._warned = True
            if self.accelerator.is_main_process:
                self.log({"cdrft/proxy_skipped": 1.0})
        finally:
            if not was_train:
                model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Soft-penalty variant (PROXY_MODE=joint with estimate_every>0)."""
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        every = int(self.cfg["estimate_every"])
        step = int(getattr(self.state, "global_step", 0))
        if every <= 0 or step % every != 0 or not self.bshared_probes or self.cfg["PROXY_MODE"] != "joint":
            return loss
        base_loss, outputs = loss if isinstance(loss, tuple) else (loss, None)
        pen, stats = self._proxy(self.bshared_probes[0])
        if self.accelerator.is_main_process:
            self.log({f"cdrft/{k}": v for k, v in stats.items()})
        if torch.is_tensor(pen) and pen.requires_grad:
            total = base_loss + pen.to(base_loss.dtype)
            return (total, outputs) if return_outputs else total
        return loss
