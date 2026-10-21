# Scaling Study (Nano-AI, track 7)

MLP de siguiente palabra en NumPy puro, entrenado en 4 tamaños (16/32/64/128 neuronas ocultas)
y medido en una Raspberry Pi. **El entregable es la curva de escalado**, no el autocompletador.
Lee `README.md` para estructura y comandos.

## Reglas que no se rompen
- Entre tamaños SOLO cambia `hidden`. Datos, épocas, optimizador y semilla viven en `src/config.py`;
  si se cambia algo ahí, se reentrenan y reevalúan los 4 tamaños.
- El test set es fijo (`data/clean/test.sha256`). No usar nada del test para entrenar, elegir el vocabulario
  ni ajustar hiperparámetros.
- Los números de `results/` solo pueden venir de corridas reales del equipo. Nunca inventar ni estimar números;
  nunca commitear resultados de pruebas con datos de juguete.
- El benchmark de velocidad y memoria debe correr en la Raspberry Pi, no en una laptop.
- Prohibido afirmar que el modelo "entiende" o "razona": solo autocompleta.
- Solo NumPy en `src/model.py` y `pi/` (la Pi debe correr sin instalaciones pesadas).

## Comandos
- `python tests/smoke_test.py` — verifica el pipeline con un corpus de juguete (no toca datos reales)
- `python src/prepare_data.py` → `python src/run_all.py` → `python src/plot_curves.py`
- En la Pi: `pi/run_benchmark.sh`, `python3 pi/demo.py`

## Datos
Libros de dominio público en `data/raw/train/`; test de otro autor + texto propio en `data/raw/test/`.
Cada fuente se documenta en `data/SOURCES.md`. Sin mensajes privados ni datos personales.
