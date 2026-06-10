# ARC Prize 2026 - ARC-AGI-2 code submission ("Failure as Signal", author: Si Fan)
# Purely symbolic, knowledge-free, CPU-only, offline. Writes /kaggle/working/submission.json.
import sys, os, glob, zipfile, subprocess

hits = glob.glob("/kaggle/input/**/kaggle_submit.py", recursive=True)
code_dir = os.path.dirname(hits[0]) if hits else None
if code_dir is None:
    zips = glob.glob("/kaggle/input/**/code.zip", recursive=True)
    if zips:
        with zipfile.ZipFile(zips[0]) as f:
            f.extractall("/kaggle/working/code_x")
        h2 = glob.glob("/kaggle/working/code_x/**/kaggle_submit.py", recursive=True)
        code_dir = os.path.dirname(h2[0]) if h2 else None
assert code_dir, "code not found; inputs=" + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
print("code_dir:", code_dir)

r = subprocess.run([sys.executable, os.path.join(code_dir, "kaggle_submit.py")],
                   capture_output=True, text=True, cwd=code_dir)
print(r.stdout[-3000:])
print(r.stderr[-2000:])
assert os.path.exists("/kaggle/working/submission.json"), "submission.json not produced"
print("OK: /kaggle/working/submission.json ready")
