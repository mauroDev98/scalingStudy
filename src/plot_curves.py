"""Grafica la curva de escalado: precisión, tiempo y memoria contra el tamaño del modelo.

Uso:  python src/plot_curves.py      (necesita matplotlib; genera results/curvas.png)
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS


def read(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    res, pi = read(RESULTS / "resultados.csv"), read(RESULTS / "pi_benchmark.csv")
    if not res:
        raise SystemExit("Falta results/resultados.csv: corre src/run_all.py primero.")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    p = [int(r["params"]) for r in res]
    ax[0].plot(p, [float(r["test_top3"]) for r in res], "o-", label="modelo")
    ax[0].axhline(float(res[0]["baseline_top3"]), ls="--", color="gray", label="baseline (3 más frecuentes)")
    ax[0].set(xscale="log", xlabel="parámetros", ylabel="acierto top-3 en test", title="Precisión")
    ax[0].legend()

    if pi:
        pp = [int(r["params"]) for r in pi]
        ax[1].plot(pp, [float(r["t_median_ms"]) for r in pi], "o-", color="C1")
        ax[1].set(xscale="log", xlabel="parámetros", ylabel="ms por predicción (mediana, en la Pi)", title="Velocidad")
        ax[2].plot(pp, [float(r["peak_rss_mb"]) for r in pi], "o-", color="C2")
        ax[2].set(xscale="log", xlabel="parámetros", ylabel="memoria pico (MB, en la Pi)", title="Memoria")
    else:
        for a in ax[1:]:
            a.text(0.5, 0.5, "Falta results/pi_benchmark.csv\n(correr pi/run_benchmark.sh en la Pi)", ha="center")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "curvas.png"
    fig.savefig(out, dpi=150)
    print("guardado", out)


if __name__ == "__main__":
    main()
