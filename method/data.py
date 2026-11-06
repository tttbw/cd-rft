"""Training mixture (math / code / logic), few-shot demonstrations, and the fixed control probe.

Expected files under $CDRFT_DATA (default: <repo>/data, see data/prepare_data.py):
  math.jsonl, code.jsonl, logic.jsonl        training pools
  probe/math_train.jsonl                     MATH train split (problem, solution)
  probe/mbpp_train.jsonl                     MBPP train split (text, code, test_list)
"""
import json
import os
import random

DATA = os.environ.get("CDRFT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

_R1_THINK = os.environ.get("R1_THINK", "0") == "1"
SYSTEM = ("Put your final answer within <answer> </answer> tags." if _R1_THINK
          else "Please reason step by step, and put your final answer within \\boxed{}.")
LOGIC_SYSTEM = SYSTEM
CODE_SYSTEM = ("You are an expert Python programmer. Write a correct Python solution for the task: "
               "define the required function, or read input from stdin and print to stdout, exactly as "
               "the problem specifies. Put your final solution in a single ```python ... ``` code block.")

_SCHEMA_EMPTY = {"gt": "", "test_list": [], "test_setup_code": "", "ground_truth": "",
                 "logic_gold": "", "code_functional": "", "code_io": ""}


def _ex(prompt_msgs, task, **fields):
    e = {"prompt": prompt_msgs, "task": task, **_SCHEMA_EMPTY}
    e.update(fields)
    return e


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _load_math():
    return [_ex([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": d["question"].strip()}], "math", gt=str(d["answer"]))
            for d in _jsonl(os.path.join(DATA, "math.jsonl"))]


def _load_code():
    stem = os.environ.get("CODE_DATA", "")
    rows = []
    for d in _jsonl(os.path.join(DATA, f"code_{stem}.jsonl" if stem else "code.jsonl")):
        e = _ex(d["prompt_msgs"], "code")
        if d.get("ctype") == "functional":
            e["code_functional"] = d["functional_test"]
        else:
            e["code_io"] = json.dumps({"inputs": d.get("inputs", []), "outputs": d.get("outputs", [])})
        rows.append(e)
    return rows


def _load_logic():
    rows = []
    for d in _jsonl(os.path.join(DATA, "logic.jsonl")):
        gold = d["gold"]
        rows.append(_ex(d["prompt_msgs"], "logic",
                        logic_gold=(gold if isinstance(gold, str) else json.dumps(gold))))
    return rows


FEWSHOT_DEMOS = {
    "math": [
        {"q": "What is 24 multiplied by 15?",
         "a": "<think> 24 x 15 = 24 x 10 + 24 x 5 = 240 + 120 = 360. </think> <answer> 360 </answer>"},
        {"q": "Let f be a function such that f(1)=2 and f(n)=f(n-1)+2n for all n>=2. Find f(5).",
         "a": "<think> We compute step by step. f(1)=2. f(2)=f(1)+2*2=2+4=6. f(3)=f(2)+2*3=6+6=12. "
              "f(4)=f(3)+2*4=12+8=20. f(5)=f(4)+2*5=20+10=30. </think> <answer> 30 </answer>"},
    ],
    "code": [
        {"q": "Solve the following coding problem using the programming language python:\n\n"
              "Return the number of vowels (a, e, i, o, u) in the given lowercase string.\n\n"
              "Write your solution by modifying this code:\n```python\ndef count_vowels(s):\n    \n```\n"
              "Your solution should be implemented in the function \"count_vowels\".\n\n"
              "Now solve the problem and return the code.",
         "a": "<think> Count the characters that are in the vowel set 'aeiou'. </think>\n"
              "```python\ndef count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n```"},
        {"q": "Solve the following coding problem using the programming language python:\n\n"
              "The input contains two integers a and b on one line separated by a space. Print their sum.\n\n"
              "The input will be given via stdin and the output should be printed to stdout.\n\n"
              "Now solve the problem and return the code.",
         "a": "<think> Read the line from stdin, split into two integers, print their sum. </think>\n"
              "```python\na, b = map(int, input().split())\nprint(a + b)\n```"},
        {"q": "You are an expert Python programmer. You will be given a question (problem specification) "
              "and will generate a correct Python program that matches the specification and passes all "
              "tests.\n\n### Question:\nGiven an integer array nums, return the second largest distinct "
              "value in it. It is guaranteed that at least two distinct values exist.\n\n \nConstraints:\n\n"
              "2 <= nums.length <= 1000\n\n\n### Format:\nYou will use the following starter code to write "
              "the solution to the problem and enclose your code within delimiters.\n```python\nclass "
              "Solution:\n    def secondLargest(self, nums: List[int]) -> int:\n        \n```",
         "a": "<think> Values may repeat, so first take the distinct values with set(nums). Sort them in "
              "descending order; the second largest is then the element at index 1. For [3,1,4,1,5] the "
              "distinct sorted-desc list is [5,4,3,1], so the answer is 4. I must keep the given class and "
              "method signature. </think>\n```python\nclass Solution:\n    def secondLargest(self, nums: "
              "List[int]) -> int:\n        return sorted(set(nums), reverse=True)[1]\n```"},
        {"q": "Solve the following coding problem using the programming language python:\n\n"
              "The first line contains an integer n. The second line contains n space-separated integers. "
              "Print the largest of them.\n\nThe input will be given via stdin and the output should be "
              "printed to stdout.\n\nNow solve the problem and return the code.",
         "a": "<think> Read n, then read the list of n integers, and print the maximum. </think>\n"
              "```python\nn = int(input())\nnums = list(map(int, input().split()))\nprint(max(nums))\n```"},
    ],
    "logic": [
        {"q": "Solve the following puzzle to determine the order of the birds from left to right. "
              "The birds are robin, hawk, dove.. The constraints are: 1. The dove is to the right of the hawk\n"
              "2. There is 1 bird between the robin and the dove\n. Please put your answer within <answer> "
              "and </answer> tags, for example <answer> ['pigeon', 'sparrow', 'quail'] </answer>.",
         "a": "<think> There are 3 positions from left to right. Constraint 2 puts exactly one bird between "
              "robin and dove, so they occupy positions 1 and 3 and the hawk is in the middle (position 2). "
              "Constraint 1 says the dove is to the right of the hawk, so the dove is at position 3 and the "
              "robin at position 1. Order: robin, hawk, dove. </think> <answer> ['robin', 'hawk', 'dove'] </answer>"},
        {"q": "Given the following list of predicates: Given Mary Love is idin then Laura Colon is jedlab. "
              "Mary Love is idin implies Eric Cook is ohawa. If Mary Love is idin, then Colin Hughes is nidi. "
              "Kathy Fisher is yutare is true if Mary Love is idin.\n\nIf Mary Love is idin, what is the next "
              "step to prove that Laura Colon is jedlab? Respond with only the trait of the next step. Please "
              "put your answer within <answer> and </answer> tags, for example <answer> fdebme </answer>.",
         "a": "<think> We are given that Mary Love is idin. The predicate 'Given Mary Love is idin then Laura "
              "Colon is jedlab' fires directly, so the next step is that Laura Colon is jedlab, i.e. the trait "
              "jedlab. </think> <answer> jedlab </answer>"},
    ],
}


def _inject_fewshot(prompt_msgs, task, k):
    demos = FEWSHOT_DEMOS.get(task, [])[:k]
    if not demos:
        return prompt_msgs
    system = [m for m in prompt_msgs if m.get("role") == "system"]
    rest = [m for m in prompt_msgs if m.get("role") != "system"]
    shots = []
    for d in demos:
        shots.append({"role": "user", "content": d["q"]})
        shots.append({"role": "assistant", "content": d["a"]})
    return system + shots + rest


def load_multi3(families, per_family, fewshot=None, seed=0, max_examples=None):
    """Balanced mixture: each family resampled to `per_family` prompts, then shuffled.
    fewshot: optional {family: number of demonstrations}."""
    import datasets
    loaders = {"math": _load_math, "code": _load_code, "logic": _load_logic}
    rng = random.Random(seed)
    all_rows = []
    for fam in families:
        rows = loaders[fam]()
        rng.shuffle(rows)
        if len(rows) >= per_family:
            rows = rows[:per_family]
        else:
            reps = (per_family + len(rows) - 1) // len(rows)
            rows = (rows * reps)[:per_family]
        all_rows.extend(rows)
    fewshot = fewshot or {}
    if any(int(fewshot.get(f, 0)) > 0 for f in families):
        for r in all_rows:
            k = int(fewshot.get(r.get("task"), 0))
            if k > 0:
                r["prompt"] = _inject_fewshot(r["prompt"], r.get("task"), k)
    ds = datasets.Dataset.from_list(all_rows).shuffle(seed=seed)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


# System prompts used by the offline control measurements (measure/control_matrix.py).
PROBE_PROMPTS = {
    "measurement": {
        "math": "Please reason step by step, and put your final answer within \\boxed{}.",
        "code": ("You are an expert Python programmer. Write a correct Python function for the task. "
                 "Put your final solution in a single ```python ... ``` code block."),
        "logic": "Please reason step by step, and put your final answer within \\boxed{}.",
    },
}


def build_bshared_probe(tokenizer, n_per_family, seed, max_len, families=None, prompt_version=None):
    """Fixed probe of reference targets y*: list of {input_ids, target_mask, task}.

    math: MATH train solutions; code: MBPP train reference programs; logic: gold answers of the
    logic pool. `seed` selects the probe samples; `prompt_version=None` uses the training system prompts.
    """
    rng = random.Random(seed)
    items = []
    allow = set(families) if families else {"math", "code", "logic"}
    if prompt_version:
        ps = PROBE_PROMPTS[prompt_version]
        sys_math, sys_code, sys_logic = ps["math"], ps["code"], ps["logic"]
    else:
        sys_math, sys_code, sys_logic = SYSTEM, CODE_SYSTEM, LOGIC_SYSTEM

    def encode(system, user, target, task):
        if getattr(tokenizer, "chat_template", None) is None:
            prompt_ids = list(tokenizer(f"{system}\n\n{user}\n\nAnswer:", add_special_tokens=True)["input_ids"])
        else:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            prompt_ids = list(prompt_ids)
        tgt_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
        ids = (prompt_ids + tgt_ids)[:max_len]
        mask = ([0] * len(prompt_ids) + [1] * len(tgt_ids))[:max_len]
        if sum(mask) < 2:
            return None
        return {"input_ids": ids, "target_mask": mask, "task": task}

    def take(rows, fn, task):
        rng.shuffle(rows)
        for r in rows:
            e = fn(r)
            if e:
                items.append(e)
            if sum(1 for x in items if x["task"] == task) >= n_per_family:
                break

    if "math" in allow:
        take(_jsonl(os.path.join(DATA, "probe", "math_train.jsonl")),
             lambda r: encode(sys_math, r["problem"].strip(), " " + r["solution"].strip(), "math"), "math")
    if "code" in allow:
        def code_item(r):
            tests = "\n".join(r["test_list"])
            user = f"{r['text'].strip()}\n\nYour function must pass these tests:\n{tests}"
            return encode(sys_code, user, "\n```python\n" + r["code"].strip() + "\n```", "code")
        take(_jsonl(os.path.join(DATA, "probe", "mbpp_train.jsonl")), code_item, "code")
    if "logic" in allow:
        def logic_item(r):
            gold = r["gold"]
            if not isinstance(gold, str):
                try:
                    gold = json.loads(gold)
                except Exception:
                    pass
            gold_s = ", ".join(map(str, gold)) if isinstance(gold, (list, tuple)) else str(gold)
            return encode(sys_logic, r["prompt_msgs"][-1]["content"],
                          " The answer is \\boxed{" + gold_s + "}.", "logic")
        take(_jsonl(os.path.join(DATA, "logic.jsonl")), logic_item, "logic")
    return items
