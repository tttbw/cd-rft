"""Per-step number of projection updates of a CD-RFT run: {step: n} for steps 1..max_step."""
import json
import sys


def main(log_path, out_path, max_step=1000):
    sched = {}
    for line in open(log_path):
        r = json.loads(line)
        if "cdrft/proxy_inner_iters" in r and "step" in r and r["step"] <= max_step:
            sched[int(r["step"])] = int(r["cdrft/proxy_inner_iters"])
    if max(sched, default=0) >= max_step:
        for s in range(1, max_step + 1):
            sched.setdefault(s, 0)
    json.dump({str(k): v for k, v in sorted(sched.items())}, open(out_path, "w"), indent=0)
    print(f"{out_path}: {sum(sched.values())} updates over {len(sched)} steps")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
