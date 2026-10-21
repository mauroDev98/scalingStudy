"""Entrena y evalúa todos los tamaños con la MISMA configuración (src/config.py).

Uso:  python src/run_all.py            # 16, 32, 64, 128
      python src/run_all.py --sizes 16 32
"""
import argparse

from config import SIZES
from evaluate import evaluate
from train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    for h in ap.parse_args().sizes:
        print(train(h))
        r = evaluate(h)
        print(f"==> h={h}: test top-3 {r['test_top3']:.1%} (baseline {r['baseline_top3']:.1%})\n", flush=True)
