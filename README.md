# Scaling Study — Autocompletado de siguiente palabra en una Raspberry Pi

Nano-AI, **track 7 (Scaling Study)**. Entrenamos el **mismo** MLP pequeño en 4 tamaños
(16, 32, 64 y 128 neuronas ocultas) y medimos cada uno: acierto en un test fijo, tiempo por
predicción y memoria en la Raspberry Pi (y energía, si hay medidor USB).

**El resultado del proyecto es la curva: ¿cuánto mejora el modelo al duplicar su tamaño?**
El autocompletador es solo la tarea sobre la que medimos.

## La tarea
Dadas las 3 palabras anteriores, sugerir 3 palabras para la siguiente. Métrica: **acierto top-3**
(la palabra correcta está entre las 3 sugerencias) sobre texto de autores que el modelo nunca vio.
Baseline: sugerir siempre las 3 palabras más frecuentes del entrenamiento.

## Reglas del estudio (no negociables)
1. **Un solo dataset y un solo test set congelado** para todos los tamaños. `prepare_data.py` guarda la huella del
   test en `data/clean/test.sha256` y se detiene si el test cambia.
2. **Solo cambia el ancho de la capa oculta.** Datos, épocas, optimizador y semilla viven en `src/config.py`.
   Si cambian algo ahí, hay que reentrenar y reevaluar los 4 tamaños.
3. **Todo número reportado sale del test** (texto que el modelo no vio) y **todo modelo parte de pesos aleatorios**.
4. **No decimos que entiende nada.** Solo autocompleta con patrones estadísticos de sus textos.
5. Nada va al reporte si no se produjo en vivo o quedó registrado en un archivo de `results/`.

## Estructura
```
data/raw/train/    libros de entrenamiento (.txt, dominio público)
data/raw/test/     otro autor + texto escrito por el equipo (NUNCA entrena)
data/clean/        salida de prepare_data.py (vocab, stats, test.npz, test.sha256)
data/SOURCES.md    de dónde salió cada texto y qué se quitó   <- lo llena el Data lead
src/               config, prepare_data, model, train, evaluate, run_all, plot_curves
pi/                benchmark.py, run_benchmark.sh, demo.py (corren en la Raspberry Pi)
models/            pesos entrenados (.npz) — se versionan para llevarlos a la Pi
results/           resultados.csv, pi_benchmark.csv, curves/, curvas.png, demo_log.jsonl
report/            plantilla del reporte de 2 páginas y bitácoras
docs/              plan de 2 semanas (Excel)
tests/             prueba de humo del pipeline
```

## Cómo se corre
Solo necesita NumPy (y matplotlib para la gráfica). Sin PyTorch: el modelo está escrito en NumPy puro,
así que en la Pi no hay instalaciones complicadas.

```bash
pip install -r requirements.txt
python tests/smoke_test.py          # comprueba que el pipeline funciona (corpus de juguete, 1 segundo)

# 1) Poner los textos en data/raw/train/ y data/raw/test/ (ver data/SOURCES.md), luego:
python src/prepare_data.py          # limpia, arma vocabulario y congela el test

# 2) Entrenar y evaluar los 4 tamaños (misma configuración)
python src/run_all.py               # o uno solo: python src/train.py --hidden 16 && python src/evaluate.py --hidden 16

# 3) En la Raspberry Pi: medir velocidad, memoria y energía
pi/run_benchmark.sh --label pi4-4GB                      # sin medidor
pi/run_benchmark.sh --watts 3.9 --idle-watts 2.7         # con medidor USB (potencia promedio y en reposo)

# 4) Curvas y demo
python src/plot_curves.py           # results/curvas.png
python3 pi/demo.py                  # demo en vivo: sugerencias de cada tamaño, queda registrada
```
Tiempos de referencia: con ~400,000 palabras de entrenamiento, los 4 tamaños tardan unos 5 minutos en total
en un servidor. **En una Pi espera varias veces más**; si el instructor exige entrenar ahí, empiecen pronto.
La GPU de la Pi no se usa: no es compatible con los frameworks de entrenamiento comunes.

## Qué hace cada rol primero
| Rol | Primera tarea |
|---|---|
| Data lead | Elegir 2 a 3 libros de entrenamiento y 1 de test de OTRO autor; escribir el texto propio; llenar `data/SOURCES.md` |
| Model lead | Correr `tests/smoke_test.py`, leer `src/model.py` y poder explicar el modelo y la curva de pérdida |
| Pi y tools lead | Preparar la Pi (Python 3 + NumPy), clonar este repo, correr `pi/demo.py` con cualquier modelo y `pi/run_benchmark.sh` |
| Evaluación y reporte lead | Estudiar las métricas de `src/evaluate.py`, preparar el reporte desde `report/reporte_TEMPLATE.md` |

Cada integrante lleva su bitácora semanal en `report/bitacora_<nombre>.md` (plantilla incluida).

## Checklist de entrega
- [ ] Mismo modelo en 4 o más tamaños, mismo dataset y test fijo
- [ ] `results/resultados.csv` con las 4 filas (acierto, pérdida, baseline)
- [ ] `results/pi_benchmark.csv` medido en la Pi (tiempo y memoria; energía si hay medidor)
- [ ] Gráfica de la curva (`results/curvas.png`)
- [ ] Demo en vivo funcionando en la Pi
- [ ] Dataset (`data/raw` + `data/SOURCES.md`) y código en este repo
- [ ] Reporte de 2 páginas con nombres (`report/`)
- [ ] Bitácora de cada integrante

## Equipo
| Rol | Integrante |
|---|---|
| Data lead | |
| Model lead | |
| Pi y tools lead | |
| Evaluación y reporte lead | |
<img width="479" height="640" alt="imagen" src="https://github.com/user-attachments/assets/372d64c9-cbb8-4914-85f4-3d89b3844e03" />
