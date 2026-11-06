"""Run configuration: DEFAULTS, optionally overridden by a JSON file passed with --config."""
import inspect
import json

REWARD_CORRECT = 3.0
REWARD_WRONG = -0.5

DEFAULTS = {
    "learning_rate": 2e-6,
    "max_steps": 1000,
    "num_generations": 4,
    "max_completion_length": 512,
    "temperature": 0.6,
    "kl_beta": 0.0,
    "lambda2": 1.0,
    "PROXY_TARGET": 0.70,
    "PROXY_KMAX": 12,
    "PROXY_LR": 4e-3,
    "PROXY_FD_EPS": 0.05,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "max_prompt_length": 512,
    "per_family": 3000,
    "fewshot": None,
    "mask_truncated": None,
    "save_steps": 50,
    "snapshot_steps": 0,
    "stop_at_step": 0,
    "gate_level": "sublayer",
    "estimate_every": 0,
    "PROXY_MODE": "indep",
    "PROXY_FD_SAMPLES": 2,
    "PROXY_DENOM_SHIFT": 1e-4,
    "probe_per_family": 3,
    "probe_max_len": 448,
    "probe_seeds": 1,
}

METHOD_DEFAULTS = {
    "grpo": {"per_device_batch": 4, "grad_accum": 8},
    "cdrft": {"per_device_batch": 1, "grad_accum": 32},
}

METHOD_KNOBS = {
    "grpo": dict(loss_type="grpo", scale_rewards=True),
    "cdrft": dict(loss_type="grpo", scale_rewards=True),
}


def load_config(path, method):
    cfg = {}
    if path:
        with open(path) as f:
            cfg = json.load(f)
    for k, v in {**DEFAULTS, **METHOD_DEFAULTS[method]}.items():
        if cfg.get(k) is None:
            cfg[k] = v
    return cfg


def filter_supported(cls, kwargs: dict) -> dict:
    """Keep only the keyword arguments accepted by the installed version of `cls`."""
    try:
        sig = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return kwargs
    keep = {k: v for k, v in kwargs.items() if k in sig}
    dropped = set(kwargs) - set(keep)
    if dropped:
        print(f"[configs] dropping unsupported {cls.__name__} kwargs: {sorted(dropped)}")
    return keep
