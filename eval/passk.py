"""Per-sample correctness, the unbiased pass@k estimator, and paired seed-level aggregation.

Generations are sampled with vLLM (n samples per prompt, settings in the paper appendix); each record
holds the per-prompt number of correct samples `n_correct` out of `n`.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "method"))

DOMAINS = {
    "math": ["MATH500", "AMC23", "AIME24", "AIME25", "Minerva"],
    "code": ["HumanEval+", "MBPP"],
    "logic": ["GURU-ordering", "GURU-graph"],
}
HARD = ["AIME24", "AIME25", "Minerva"]
K_REPORT = {"AIME24": 256, "AIME25": 256, "Minerva": 256, "AMC23": 64, "HumanEval+": 64, "MBPP": 64,
            "GURU-graph": 64, "MATH500": 16, "GURU-ordering": 16}


def is_correct(family, text, item, timeout=3.0):
    """family in {math, humaneval, mbpp, logic}; `item` carries the gold answer or tests."""
    if family == "math":
        from reward import DATA_NAME, _answer_tag_or_text, _math_equal_worker, _with_timeout, extract_answer
        gold = extract_answer("\\boxed{%s}" % item["gold"], DATA_NAME, use_last_number=True)
        pred = extract_answer(_answer_tag_or_text(text), DATA_NAME, use_last_number=True)
        return _with_timeout("equal", _math_equal_worker, gold, pred, timeout)
    if family == "mbpp":
        from code_reward import REWARD_CORRECT, score_one_code
        return score_one_code(text, item["test_list"], item.get("setup", "")) >= REWARD_CORRECT - 1e-6
    if family == "humaneval":
        from code_reward import extract_code
        harness = f"{extract_code(text)}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(harness)
                p = f.name
            r = subprocess.run([sys.executable, p], capture_output=True, timeout=15, cwd=tempfile.gettempdir())
            os.unlink(p)
            return r.returncode == 0
        except Exception:
            return False
    if family == "logic":
        from logic_reward import score_one_logic
        return score_one_logic(text, item["gold"]) >= 2.0
    raise ValueError(family)


def pass_at_k(n, c, k):
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def per_prompt(records, k):
    """records: [{'n': int, 'n_correct': int}] -> (pass@1 array, pass@k array); k per benchmark: K_REPORT."""
    p1 = np.array([r["n_correct"] / r["n"] for r in records], dtype=float)
    pk = np.array([pass_at_k(r["n"], r["n_correct"], k) for r in records], dtype=float)
    return p1, pk


def aggregate(bench_means, hard_pool=None):
    """Benchmark means -> domain means (equal weights) and the overall mean of the three domains."""
    dom = {d: float(np.mean([bench_means[b] for b in bs])) for d, bs in DOMAINS.items()}
    out = {"overall": float(np.mean(list(dom.values()))), **dom,
           "hard": float(np.mean([bench_means[h] for h in HARD]))}
    if hard_pool is not None:
        out["hard_pool"] = float(np.mean(hard_pool))
    return out


def paired_bootstrap(per_seed, B=2000, seed=0):
    """Stratified bootstrap of paired differences.

    per_seed: one dict per training seed, {benchmark: per-prompt difference array (method - baseline)},
    with prompts aligned between the two arms. Seeds are resampled with replacement, then prompts
    within each drawn seed. Returns {column: (point estimate, 2.5th, 97.5th percentile)}.
    """
    rng = np.random.default_rng(seed)

    def stats(d, draw):
        m = {}
        for b, v in d.items():
            m[b] = float(v[rng.integers(0, len(v), len(v))].mean() if draw else v.mean())
        hp = np.concatenate([d[h] for h in HARD])
        return aggregate(m, hp[rng.integers(0, len(hp), len(hp))] if draw else hp)

    points = [stats(d, False) for d in per_seed]
    cols = list(points[0])
    point = {c: float(np.mean([p[c] for p in points])) for c in cols}
    res = {c: np.empty(B) for c in cols}
    S = len(per_seed)
    for r in range(B):
        picks = rng.integers(0, S, S) if S > 1 else [0]
        acc = {c: 0.0 for c in cols}
        for s in picks:
            v = stats(per_seed[s], True)
            for c in cols:
                acc[c] += v[c]
        for c in cols:
            res[c][r] = acc[c] / len(picks)
    return {c: (point[c], *np.percentile(res[c], [2.5, 97.5])) for c in cols}
