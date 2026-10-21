#!/usr/bin/env bash
# Mide todos los tamaños en la Pi, cada uno en un proceso nuevo.
# Uso:  pi/run_benchmark.sh                      (tamaños por defecto)
#       SIZES="16 32 64 128" pi/run_benchmark.sh --label pi4-2GB
#       pi/run_benchmark.sh --watts 3.9 --idle-watts 2.7
set -euo pipefail
cd "$(dirname "$0")/.."
for h in ${SIZES:-16 32 64 128}; do
  python3 pi/benchmark.py --hidden "$h" "$@"
done
