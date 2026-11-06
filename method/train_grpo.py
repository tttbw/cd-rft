"""Training entry point: GRPO or CD-RFT on the math / code / logic mixture.

  python method/train_grpo.py --method cdrft --seed 0 --output_dir runs/cdrft [--config overrides.json]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from configs import METHOD_KNOBS, filter_supported, load_config
from data import build_bshared_probe, load_multi3
from reward import make_router_reward_funcs


class LogHistoryCallback(TrainerCallback):
    def __init__(self, path):
        self.path = path

    def _dump(self, state):
        with open(self.path, "w") as f:
            for rec in state.log_history:
                f.write(json.dumps(rec) + "\n")

    def on_save(self, args, state, control, **kw):
        if state.is_world_process_zero:
            self._dump(state)

    def on_train_end(self, args, state, control, **kw):
        if state.is_world_process_zero:
            self._dump(state)


class SnapshotCallback(TrainerCallback):
    """Weights-only snapshots every `every` steps under <output_dir>/snapshots/step-<N>/."""

    def __init__(self, model, tokenizer, output_dir, every):
        self.model, self.tok, self.every = model, tokenizer, every
        self.dir = os.path.join(output_dir, "snapshots")

    def on_step_end(self, args, state, control, **kw):
        s = int(state.global_step)
        if self.every > 0 and s > 0 and s % self.every == 0 and state.is_world_process_zero:
            d = os.path.join(self.dir, f"step-{s}")
            os.makedirs(d, exist_ok=True)
            self.model.save_pretrained(d, safe_serialization=True)
            self.tok.save_pretrained(d)


class StopAtCallback(TrainerCallback):
    """Stop at a given step while the learning-rate schedule still follows max_steps."""

    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kw):
        if state.global_step >= self.step:
            control.should_training_stop = True


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON file overriding configs.DEFAULTS")
    ap.add_argument("--method", choices=["grpo", "cdrft"], default="cdrft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--families", nargs="+", default=["math", "code", "logic"],
                    choices=["math", "code", "logic"])
    ap.add_argument("--placebo", choices=["perm", "rand"], default=None)
    ap.add_argument("--dose_schedule", default=None)
    ap.add_argument("--optim8bit", action="store_true")
    ap.add_argument("--vllm_sleep", action="store_true")
    ap.add_argument("--no_vllm", dest="use_vllm", action="store_false")
    return ap.parse_args()


def main():
    a = build_args()
    c = load_config(a.config, a.method)
    env = os.environ.get

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if env("R1_THINK", "0") == "1":
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates",
                               "r1_think.jinja")) as f:
            tok.chat_template = f.read()
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16,
                                                 attn_implementation=attn_impl)

    ds = load_multi3(families=tuple(a.families), per_family=c["per_family"], fewshot=c["fewshot"])

    cfg = dict(
        output_dir=a.output_dir,
        learning_rate=c["learning_rate"], lr_scheduler_type=c["lr_scheduler_type"],
        warmup_ratio=c["warmup_ratio"], weight_decay=c["weight_decay"],
        adam_beta1=c["adam_beta1"], adam_beta2=c["adam_beta2"], max_grad_norm=c["max_grad_norm"],
        per_device_train_batch_size=c["per_device_batch"], gradient_accumulation_steps=c["grad_accum"],
        num_generations=c["num_generations"], max_completion_length=c["max_completion_length"],
        max_prompt_length=c["max_prompt_length"], temperature=c["temperature"],
        max_steps=c["max_steps"], num_train_epochs=1.0, beta=c["kl_beta"],
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_steps=int(c["save_steps"]), save_total_limit=1, seed=a.seed,
        use_vllm=a.use_vllm, vllm_mode="colocate", vllm_tensor_parallel_size=1,
        vllm_enable_sleep_mode=a.vllm_sleep,
        vllm_gpu_memory_utilization=float(env("VLLM_UTIL", "0.40")),
        vllm_max_model_length=int(env("VLLM_MAX_LEN", "4096")),
        report_to=["none"],
    )
    cfg.update(METHOD_KNOBS[a.method])
    if c["mask_truncated"] is not None:
        cfg["mask_truncated_completions"] = bool(c["mask_truncated"])
    if a.optim8bit:
        cfg["optim"] = "paged_adamw_8bit"
    args = GRPOConfig(**filter_supported(GRPOConfig, cfg))

    reward_funcs = make_router_reward_funcs(families=a.families)

    if a.method == "cdrft":
        from cd_rft import CDRFTTrainer
        probes = [build_bshared_probe(tok, n_per_family=c["probe_per_family"], seed=s,
                                      max_len=c["probe_max_len"], families=tuple(a.families))
                  for s in range(int(c["probe_seeds"]))]
        trainer = CDRFTTrainer(model=model, reward_funcs=reward_funcs, args=args, train_dataset=ds,
                               processing_class=tok, cdrft_cfg=c, bshared_probes=probes, seed=a.seed,
                               placebo=a.placebo, dose_schedule=a.dose_schedule)
    else:
        trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=args, train_dataset=ds,
                              processing_class=tok)

    os.makedirs(a.output_dir, exist_ok=True)
    trainer.add_callback(LogHistoryCallback(os.path.join(a.output_dir, "log_history.jsonl")))
    if c["snapshot_steps"]:
        trainer.add_callback(SnapshotCallback(model, tok, a.output_dir, int(c["snapshot_steps"])))
    if c["stop_at_step"]:
        trainer.add_callback(StopAtCallback(int(c["stop_at_step"])))

    ckpts = glob.glob(os.path.join(a.output_dir, "checkpoint-*"))
    resume = max(ckpts, key=lambda p: int(p.split("-")[-1])) if ckpts else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)


if __name__ == "__main__":
    main()
