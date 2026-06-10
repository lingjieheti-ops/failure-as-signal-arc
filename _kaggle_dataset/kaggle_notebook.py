# ARC Prize 2026 — ARC-AGI-2 code submission (paper-track entry "Failure as Signal", author: Si Fan)
# Purely symbolic, knowledge-free, CPU-only, offline. Expects the competition data under /kaggle/input
# and this repo's code/ attached as a Kaggle Dataset named e.g. "failure-as-signal-code".
# Runtime: ~minutes for 120 tasks (well under the 12h limit). Writes /kaggle/working/submission.json.
import sys, os, glob, shutil

# locate the attached code directory (dataset can be mounted under any name)
code_dir = None
for cand in glob.glob("/kaggle/input/*/code") + glob.glob("/kaggle/input/*/*/code"):
    if os.path.exists(os.path.join(cand, "kaggle_submit.py")):
        code_dir = cand
        break
assert code_dir, "attach the repo (with code/) as a Kaggle Dataset"
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.dirname(code_dir))

import subprocess
r = subprocess.run([sys.executable, os.path.join(code_dir, "kaggle_submit.py")],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
print(r.stderr[-2000:])
assert os.path.exists("/kaggle/working/submission.json"), "submission.json not produced"
print("OK: /kaggle/working/submission.json ready")
