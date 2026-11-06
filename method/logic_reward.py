"""Logic reward: compare the final answer with the gold sequence or string."""
import json
import re


def _extract_answer(text):
    """<answer>...</answer>, then \\boxed{...}, then the last bracketed list, then the last line."""
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"\[([^\[\]]+)\]", text)
    if m:
        return m[-1].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _tokens(s):
    s = re.sub(r"[\[\]'\"]", " ", str(s))
    return [t for t in re.split(r"[,\s]+", s.strip().lower()) if t]


def score_one_logic(completion_text, gold, full=3.0, none=-0.5):
    try:
        if isinstance(gold, str):
            try:
                g = json.loads(gold)
            except Exception:
                g = gold
        else:
            g = gold
        gold_toks = _tokens(g if not isinstance(g, (list, tuple)) else " ".join(map(str, g)))
        if not gold_toks:
            return 0.0
        ans_toks = _tokens(_extract_answer(completion_text))
        if ans_toks == gold_toks:
            return full
        if len(ans_toks) >= len(gold_toks) and ans_toks[-len(gold_toks):] == gold_toks:
            return full
        if set(gold_toks) and set(gold_toks) <= set(ans_toks):
            return 0.5
        return none
    except Exception:
        return none
