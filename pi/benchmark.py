"""Mide en la Raspberry Pi UN tamaño de modelo: tiempo por predicción, memoria y (opcional) energía.

Uso:  python3 pi/benchmark.py --hidden 64
      python3 pi/benchmark.py --hidden 64 --watts 3.9 --idle-watts 2.7    (con medidor USB)

Corre cada tamaño en un proceso NUEVO (pi/run_benchmark.sh lo hace) para que la memoria pico
de un modelo no contamine a otro. Fuerza 1 hilo de BLAS: la latencia de una sola predicción es
más estable así. Las predicciones usan contextos aleatorios: el cómputo es el mismo que con texto real.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CONTEXT, MODELS, RESULTS, TOPK  # noqa: E402
from model import MLP  # noqa: E402
from utils import upsert_csv  # noqa: E402

FIELDS = ["hidden", "label", "params", "model_file_kb", "rss_base_mb", "peak_rss_mb", "t_median_ms",
          "t_p95_ms", "n_preds", "watts", "idle_watts", "energy_mj", "host", "python", "numpy", "timestamp"]


def peak_rss_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 * 1024) if platform.system() == "Darwin" else kb / 1024


def host_name():
    for p in ("/proc/device-tree/model",):
        try:
            return Path(p).read_text().strip("\x00\n ")
        except OSError:
            pass
    return platform.platform()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--n", type=int, default=2000, help="número de predicciones medidas")
    ap.add_argument("--label", default="pi", help="nombre del equipo medido, p. ej. pi4-2GB")
    ap.add_argument("--watts", type=float, help="potencia promedio medida con el medidor USB durante la corrida")
    ap.add_argument("--idle-watts", type=float, default=0.0, help="potencia en reposo (se resta)")
    a = ap.parse_args()

    path = MODELS / f"mlp_h{a.hidden}.npz"
    rss_base = peak_rss_mb()
    model = MLP.load(path)
    rng = np.random.default_rng(0)
    ctx = rng.integers(1, model.vocab_size, size=(a.n + 200, CONTEXT))

    for i in range(200):  # calentamiento
        model.top_k(ctx[i:i + 1], TOPK)
    times = np.empty(a.n)
    for i in range(a.n):
        x = ctx[200 + i:201 + i]
        t0 = time.perf_counter_ns()
        model.top_k(x, TOPK)
        times[i] = (time.perf_counter_ns() - t0) / 1e6
    peak = peak_rss_mb()

    med = float(np.median(times))
    energy = round((a.watts - a.idle_watts) * med, 4) if a.watts is not None else ""  # W * ms = mJ
    row = {
        "hidden": a.hidden, "label": a.label, "params": model.n_params,
        "model_file_kb": round(path.stat().st_size / 1024, 1),
        "rss_base_mb": round(rss_base, 1), "peak_rss_mb": round(peak, 1),
        "t_median_ms": round(med, 4), "t_p95_ms": round(float(np.percentile(times, 95)), 4),
        "n_preds": a.n, "watts": a.watts if a.watts is not None else "", "idle_watts": a.idle_watts,
        "energy_mj": energy, "host": host_name(), "python": platform.python_version(),
        "numpy": np.__version__, "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    upsert_csv(RESULTS / "pi_benchmark.csv", row, key=["hidden", "label"], fields=FIELDS)
    print(f"h={a.hidden:>4}  {row['t_median_ms']} ms (p95 {row['t_p95_ms']})  pico {row['peak_rss_mb']} MB  "
          f"energía {energy or 'n/d'} mJ  [{row['host']}]")


if __name__ == "__main__":
    main()
