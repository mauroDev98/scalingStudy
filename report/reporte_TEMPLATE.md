# Scaling Study: ¿cuánto aporta duplicar un autocompletador? (título provisional)

**Equipo:** (nombres de los 4 integrantes y rol de cada uno) · **Fecha:** 

> Máximo 2 páginas. Solo números medidos por ustedes (están en `results/`). Borren estas indicaciones al terminar.

## 1. Resultado principal (3 líneas)
Entrenamos el mismo MLP desde pesos aleatorios en 4 tamaños (16, 32, 64, 128 neuronas ocultas).
En N = ___ ejemplos de test de un autor que no vio, el acierto top-3 pasó de ___% a ___%
(baseline: ___%). En una Raspberry Pi (modelo ___) el tiempo por sugerencia pasó de ___ ms a ___ ms
y la memoria pico de ___ MB a ___ MB. Duplicar el modelo aportó ___ puntos de acierto.

## 2. Datos
Fuentes (de `data/SOURCES.md`), palabras de entrenamiento / validación / test, tamaño del vocabulario,
% de palabras desconocidas, qué se quitó y por qué. Cómo se garantiza que el test no se vio al entrenar
(autores distintos, huella `test.sha256`).

## 3. Método
Arquitectura (3 palabras → embedding de 32 → capa oculta tanh → softmax), qué cambia entre tamaños
(solo el ancho de la capa oculta) y qué es idéntico (épocas, optimizador, semilla, datos). Cómo se midió
cada número: acierto top-3, tiempo por predicción (mediana de ___ predicciones, 1 hilo), memoria pico,
energía (si hubo medidor, o "no medida").

## 4. Resultados
Tabla con las 4 filas de `results/resultados.csv` + `results/pi_benchmark.csv`, y la gráfica `results/curvas.png`.

| Neuronas | Parámetros | Top-3 test | Baseline | ms/predicción (Pi) | Memoria pico (Pi) | Energía |
|---|---|---|---|---|---|---|
| 16 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |

Interpretación: dónde la curva se aplana, cuánto cuesta cada mejora en tiempo y memoria.
Mostrar también la curva de pérdida (`results/curves/`) y si hubo sobreajuste.

## 5. Qué NO prueba este estudio
- No entiende ni razona: solo sugiere palabras probables según sus textos de entrenamiento.
- Una sola semilla (o 3, si se repitió): diferencias pequeñas entre tamaños pueden ser ruido.
- Resultados válidos para este corpus y este estilo de español, no para cualquier texto.
- (Agregar lo que corresponda.)

## 6. Fallos observados
Ejemplos reales de la demo (de `results/demo_log.jsonl`) donde las sugerencias fallan y por qué creen que pasó.
