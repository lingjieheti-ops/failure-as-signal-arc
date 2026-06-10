"""End-to-end test of the offline Kaggle entry: build a hidden-test-format challenges file (test outputs
stripped), run kaggle_submit, and validate the resulting submission.json."""
import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import arc_core as ac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tasks = ac.load_tasks("evaluation")
items = sorted(tasks.items())[:12]
chal = {tid: {"train": [{"input": ac.to_ll(p["input"]), "output": ac.to_ll(p["output"])} for p in t["train"]],
              "test": [{"input": ac.to_ll(p["input"])} for p in t["test"]]} for tid, t in items}
chal_path = os.path.join(ROOT, "experiments", "_challenges_test.json")
sub_path = os.path.join(ROOT, "experiments", "_sub_test.json")
with open(chal_path, "w") as f: json.dump(chal, f)

subprocess.run([sys.executable, os.path.join(ROOT, "code", "kaggle_submit.py"),
                "--challenges", chal_path, "--out", sub_path], check=True)

sub = json.load(open(sub_path))
def valid(g):
    return isinstance(g, list) and 1 <= len(g) <= 30 and all(
        isinstance(r, list) and 1 <= len(r) <= 30 and all(isinstance(v, int) and 0 <= v <= 9 for v in r) for r in g)
errs = 0
for tid, t in items:
    if tid not in sub or len(sub[tid]) != len(t["test"]): errs += 1; continue
    for e in sub[tid]:
        if not (valid(e.get("attempt_1")) and valid(e.get("attempt_2"))): errs += 1
print(f"\nKAGGLE ENTRY E2E: {len(sub)} tasks, errors={errs} -> {'OK (offline submission valid)' if errs==0 else 'FAIL'}")
