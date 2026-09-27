# usage: python3 sweep.py out.jsonl configs.json   -> runs 4 configs at a time, 1 BLAS thread each
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
out, cfgs = sys.argv[1], json.load(open(sys.argv[2]))
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
def go(c):
    r = subprocess.run([sys.executable, "bench.py", json.dumps(c)], capture_output=True, text=True, env=env)
    line = r.stdout.strip() or json.dumps(dict(c, error=r.stderr[-800:]))
    with open(out, "a") as f: f.write(line + "\n")
    print(line[:200], flush=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(go, cfgs))
print("DONE")
