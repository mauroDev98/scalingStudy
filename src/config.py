"""Configuración ÚNICA del estudio.

Regla del Scaling Study: entre una corrida y otra SOLO cambia HIDDEN (el ancho de la capa
oculta). Todo lo demás (datos, épocas, optimizador, semilla) es idéntico para los 4 tamaños.
Si cambian algo de aquí, hay que reentrenar y reevaluar TODOS los tamaños.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

UNK = "<unk>"          # palabra desconocida (índice 0 del vocabulario)
CONTEXT = 3            # cuántas palabras previas ve el modelo
VOCAB_SIZE = 3000      # palabras más frecuentes del ENTRENAMIENTO (incluye <unk>)
EMBED_DIM = 32         # tamaño del embedding de cada palabra (fijo en todos los tamaños)
TOPK = 3               # métrica: ¿la palabra correcta está entre las 3 sugerencias?

SIZES = [16, 32, 64, 128]   # neuronas ocultas: lo único que cambia entre modelos
EPOCHS = 8
BATCH = 256
LR = 2e-3
SEED = 0
