"""Convert the public training sources into the jsonl files read by method/data.py.

  python data/prepare_data.py --deepscaler deepscaler.json --coder1 code-r1-12k.parquet \
      --guru_dir guru/train --math_train math_train.jsonl --mbpp_train mbpp_train.jsonl --out data

math : DeepScaleR (problem, answer)
code : code-r1-12k (functional tests or stdin/stdout cases)
logic: GURU logic subsets with text answers (grid-output tasks are excluded)
probe: MATH train (problem, solution) and MBPP train (text, code, test_list), copied unchanged
"""
import argparse
import glob
import json
import os
import shutil

import pandas as pd


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)}")


def prep_math(src, out):
    rows = []
    for r in json.load(open(src)):
        q, a = r.get("problem", "").strip(), r.get("answer", "")
        if q and a not in ("", None):
            rows.append({"task": "math", "question": q, "answer": str(a)})
    _write(os.path.join(out, "math.jsonl"), rows)


def prep_code(src, out):
    rows = []
    for _, r in pd.read_parquet(src).iterrows():
        try:
            gt = r["reward_model"]["ground_truth"]
            d = json.loads(gt) if isinstance(gt, str) else gt
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["prompt"]]
            if not msgs:
                continue
            if d.get("functional"):
                rows.append({"task": "code", "ctype": "functional", "prompt_msgs": msgs,
                             "functional_test": d["functional"]})
            elif d.get("inputs") and d.get("outputs"):
                rows.append({"task": "code", "ctype": "stdio", "prompt_msgs": msgs,
                             "inputs": d["inputs"], "outputs": d["outputs"]})
        except Exception:
            continue
    _write(os.path.join(out, "code.jsonl"), rows)


def _norm(g):
    return g.tolist() if hasattr(g, "tolist") else g


def prep_logic(src_dir, out):
    rows = []
    for path in sorted(glob.glob(os.path.join(src_dir, "logic__*.parquet"))):
        name = os.path.basename(path)
        if any(g in name for g in ("arcagi", "barc")):
            continue
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            try:
                msgs = [{"role": m["role"], "content": m["content"]} for m in r["prompt"]]
                if "ground_truth" in df.columns and r["ground_truth"] is not None:
                    gold = _norm(r["ground_truth"])
                else:
                    gold = _norm(r["reward_model"]["ground_truth"])
                if msgs and gold is not None:
                    rows.append({"task": "logic", "prompt_msgs": msgs, "gold": gold,
                                 "src": str(r.get("data_source", name))})
            except Exception:
                continue
    _write(os.path.join(out, "logic.jsonl"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepscaler", required=True)
    ap.add_argument("--coder1", required=True)
    ap.add_argument("--guru_dir", required=True)
    ap.add_argument("--math_train", required=True)
    ap.add_argument("--mbpp_train", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "probe"), exist_ok=True)
    prep_math(a.deepscaler, a.out)
    prep_code(a.coder1, a.out)
    prep_logic(a.guru_dir, a.out)
    shutil.copy(a.math_train, os.path.join(a.out, "probe", "math_train.jsonl"))
    shutil.copy(a.mbpp_train, os.path.join(a.out, "probe", "mbpp_train.jsonl"))


if __name__ == "__main__":
    main()
