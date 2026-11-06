"""Code reward: run the extracted program against its tests in a subprocess."""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import REWARD_CORRECT, REWARD_WRONG  # noqa: E402

# CODE_STRICT=1: +3 only when all tests pass, partial passes capped at CODE_PARTIAL_MAX.
_CODE_STRICT = os.environ.get("CODE_STRICT", "0") == "1"
_CODE_PARTIAL_MAX = float(os.environ.get("CODE_PARTIAL_MAX", "0.5"))
_SCRATCH = tempfile.mkdtemp(prefix="cdrft_code_")
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _frac_to_reward(frac):
    if not _CODE_STRICT:
        return REWARD_WRONG + frac * (REWARD_CORRECT - REWARD_WRONG)
    if frac >= 1.0 - 1e-9:
        return REWARD_CORRECT
    return REWARD_WRONG + frac * (_CODE_PARTIAL_MAX - REWARD_WRONG)


def extract_code(text: str) -> str:
    """Last fenced block; otherwise the first def/class to the end, with preceding imports."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    t = text or ""
    m = re.search(r'(?m)^[ \t]*(?:def|class)\s+\w+', t)
    if m:
        imports = re.findall(r'(?m)^[ \t]*(?:import|from)\s+\S.*$', t[:m.start()])
        code = t[m.start():]
        return (("\n".join(imports) + "\n" + code) if imports else code).strip()
    return t.strip()


def _run(path, timeout_s, stdin=None):
    return subprocess.run([sys.executable, path], input=stdin, capture_output=True, text=True,
                          timeout=timeout_s, cwd=_SCRATCH)


def score_one_code(completion_text, test_list, setup="", timeout_s=8):
    """Assert-list tests (MBPP format): reward from the fraction of passing asserts."""
    if not test_list:
        return REWARD_WRONG
    lines = [setup or "", extract_code(completion_text) or "", "__cdrft_p = 0"]
    for t in test_list:
        lines += ["try:", f"    {t}", "    __cdrft_p += 1", "except Exception:", "    pass"]
    lines += [f'print("__CDRFT_PASS__", __cdrft_p, {len(test_list)})']
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("\n".join(lines))
            path = f.name
        out = _run(path, timeout_s)
        frac = 0.0
        for ln in out.stdout.splitlines():
            if ln.startswith("__CDRFT_PASS__"):
                _, p, n = ln.split()
                frac = int(p) / max(int(n), 1)
        return _frac_to_reward(frac)
    except Exception:
        return REWARD_WRONG
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass


def score_one_code_io(completion_text, functional="", inputs=None, outputs=None, timeout_s=10):
    """Functional tests (all-or-nothing) or stdin/stdout cases (fraction passed)."""
    code = extract_code(completion_text)
    if not code:
        return REWARD_WRONG
    try:
        if functional:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(f"{code}\n\n{functional}\n")
                path = f.name
            r = _run(path, timeout_s)
            os.unlink(path)
            return REWARD_CORRECT if r.returncode == 0 else REWARD_WRONG
        if inputs and outputs:
            n = min(len(inputs), len(outputs))
            if n == 0:
                return REWARD_WRONG
            npass = 0
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                path = f.name
            try:
                for inp, exp in list(zip(inputs, outputs))[:n]:
                    try:
                        if _run(path, timeout_s, stdin=str(inp)).stdout.strip() == str(exp).strip():
                            npass += 1
                    except Exception:
                        pass
            finally:
                os.unlink(path)
            return _frac_to_reward(npass / max(n, 1))
        return REWARD_WRONG
    except Exception:
        return REWARD_WRONG
