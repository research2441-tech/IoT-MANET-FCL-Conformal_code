"""Run jobs from a text file with a fixed number of parallel workers; skips finished jobs."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
from concurrent.futures import ThreadPoolExecutor
RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
nw = int(sys.argv[2]) if len(sys.argv) > 2 else 2

def name_of(args):
    a = dict(zip(args.split()[::2], args.split()[1::2]))
    n = f"{a.get('--tag','main')}__{a.get('--dataset','CICIoT2023')}__{a.get('--method','proposed')}__s{a.get('--seed','1')}"
    if '--holdout' in a: n += f"__ho-{a['--holdout']}"
    return n

def run(args):
    n = name_of(args)
    if os.path.exists(os.path.join(RES, n + ".json")):
        return
    log = os.path.join(HERE, "logs", n + ".log")
    t = time.time()
    subprocess.run(f"python3 run.py {args} > {log} 2>&1", shell=True, cwd=HERE)
    ok = os.path.exists(os.path.join(RES, n + ".json"))
    print(("OK " if ok else "FAIL ") + n + f" {time.time()-t:.0f}s", flush=True)

with ThreadPoolExecutor(nw) as ex:
    list(ex.map(run, jobs))
print("QUEUE DONE", flush=True)
