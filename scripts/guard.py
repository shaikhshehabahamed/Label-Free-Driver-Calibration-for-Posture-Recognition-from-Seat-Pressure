"""Laptop-safety helpers.  Import this FIRST in every long-running script (before numpy / torch).

 * caps BLAS / OpenMP threads (env vars must be set before numpy is imported)
 * lowers the process priority so the machine stays responsive
 * cooldown(): pauses while the GPU is hot (the GPU temperature is the only readable thermal sensor on this laptop; it idles at ~73-76 C)
"""
import ctypes
import os
import subprocess
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")


def lower_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def cooldown(hi=82, lo=76, poll=5, max_wait=900, tag=""):
    """block while the GPU is >= hi C, until it falls to <= lo C (or max_wait s).  Returns seconds waited."""
    t = gpu_temp()
    if t is None or t < hi:
        return 0
    waited = 0
    print(f"[guard] GPU {t} C >= {hi} C: pausing until <= {lo} C {tag}", flush=True)
    while waited < max_wait:
        time.sleep(poll)
        waited += poll
        t = gpu_temp()
        if t is None or t <= lo:
            break
    print(f"[guard] resumed after {waited}s (GPU {t} C)", flush=True)
    return waited


lower_priority()
