"""Verifiable rewards on a common [-0.5, +3] scale, routed by task family.

Math answers are parsed and compared with `parser.extract_answer` and `grader.math_equal` from the
Qwen2.5-Math evaluation toolkit; set EVAL_DIR to its `evaluation/` directory.
"""
import json
import multiprocessing as mp
import os
import re
import sys

EVAL_DIR = os.environ.get("EVAL_DIR", "")
if EVAL_DIR and EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from grader import is_digit, math_equal, parse_digits  # noqa: E402
from parser import extract_answer                       # noqa: E402

from code_reward import score_one_code_io               # noqa: E402
from configs import REWARD_CORRECT, REWARD_WRONG        # noqa: E402
from logic_reward import score_one_logic                # noqa: E402

DATA_NAME = os.environ.get("REWARD_DATA_NAME", "gsm8k")
_USE_MATH_VERIFY = os.environ.get("MATH_VERIFY", "0") == "1"
_EQUAL_TIMEOUT = float(os.environ.get("MATH_EQUAL_TIMEOUT", "3"))
_VERIFY_TIMEOUT = float(os.environ.get("MATH_VERIFY_TIMEOUT", "8"))
_POOLS = {}


def _math_equal_worker(gold, pred):
    try:
        return bool(math_equal(pred, gold))
    except Exception:
        return False


def _math_verify_worker(gold, pred):
    try:
        from math_verify import parse, verify
        return bool(verify(parse("\\boxed{%s}" % gold), parse("\\boxed{%s}" % pred), timeout_seconds=5))
    except Exception:
        return False


def _with_timeout(name, fn, gold, pred, timeout):
    """Exact-string shortcut, else run fn(gold, pred) in a persistent spawn worker with a hard timeout."""
    try:
        if pred is not None and str(pred).strip() == str(gold).strip():
            return True
    except Exception:
        pass
    if _POOLS.get(name) is None:
        _POOLS[name] = mp.get_context("spawn").Pool(processes=1)
    try:
        return bool(_POOLS[name].apply_async(fn, (gold, pred)).get(timeout=timeout))
    except mp.TimeoutError:
        try:
            _POOLS[name].terminate()
            _POOLS[name].join()
        except Exception:
            pass
        _POOLS[name] = None
        return False
    except Exception:
        return False


def _closeness(pred, gt):
    if not (is_digit(pred) and is_digit(gt)):
        return None
    p, g = parse_digits(pred), parse_digits(gt)
    if p is None or g is None:
        return None
    rel = abs(p - g) / max(abs(g), 1e-8)
    return max(0.0, 1.0 - min(rel, 1.0))


def _answer_tag_or_text(text):
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return m[-1] if m else text


def score_one(completion_text, gt, partial=True):
    """Math reward: +3 if correct, else partial credit for numeric answers by relative error."""
    ans = _answer_tag_or_text(completion_text)
    if _USE_MATH_VERIFY and _with_timeout("verify", _math_verify_worker, gt, ans, _VERIFY_TIMEOUT):
        return REWARD_CORRECT
    pred = extract_answer(ans, DATA_NAME, use_last_number=True)
    if _with_timeout("equal", _math_equal_worker, gt, pred, _EQUAL_TIMEOUT):
        return REWARD_CORRECT
    if partial:
        c = _closeness(pred, gt)
        if c is not None:
            return REWARD_WRONG + (REWARD_CORRECT - REWARD_WRONG) * c
    return REWARD_WRONG


def _as_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return str(completion)


def _score(task, i, text, kw):
    if task == "code":
        cf = (kw.get("code_functional") or [""] * (i + 1))[i]
        cio = (kw.get("code_io") or [""] * (i + 1))[i]
        if cf:
            return score_one_code_io(text, functional=cf)
        if cio:
            d = json.loads(cio)
            return score_one_code_io(text, inputs=d.get("inputs"), outputs=d.get("outputs"))
        return REWARD_WRONG
    if task == "logic":
        golds = kw.get("logic_gold")
        return score_one_logic(text, golds[i] if golds is not None else "")
    gts = kw.get("gt")
    return score_one(text, str(gts[i] if gts is not None else ""))


def make_router_reward_funcs(families=("math", "code", "logic")):
    """One reward function per family; each scores its own family and returns None for the others."""
    def make(fam):
        def fam_reward(prompts=None, completions=None, **kwargs):
            tasks = kwargs.get("task")
            return [(_score(fam, i, _as_text(c), kwargs)
                     if (tasks[i] if tasks is not None else "math") == fam else None)
                    for i, c in enumerate(completions or [])]
        fam_reward.__name__ = f"reward_{fam}"
        return fam_reward
    return [make(f) for f in families]
