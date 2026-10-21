# Fuentes del dataset (lo llena el Data lead)

El Data lead debe poder explicar de dónde salió cada fila y qué se quitó.

## Qué poner en `data/raw/`
| Carpeta | Contenido | Tamaño sugerido |
|---|---|---|
| `train/` | 2 a 3 libros en español de **dominio público** (.txt, UTF-8) | 300,000 palabras o más en total |
| `test/` | (a) un libro de **otro autor** que no esté en `train/`; (b) texto escrito por el equipo | (a) 20,000 palabras o más; (b) ideal 5,000 entre los 4 |

Reglas:
- El test nunca comparte autor con el entrenamiento: la métrica debe medir texto de un escritor que el modelo no vio.
- Solo texto de dominio público o escrito por ustedes. Sin mensajes privados ni datos personales de otras personas.
- Elijan autores fallecidos hace más de 100 años (en México los derechos de autor duran vida + 100 años;
  confirmen con el instructor). Project Gutenberg (gutenberg.org) tiene muchas obras en español.
- Si el archivo viene de Project Gutenberg, `prepare_data.py` quita solo el encabezado y la licencia
  de Gutenberg (entre `*** START OF` y `*** END OF`). Borren a mano prólogos, notas o índices que no sean del autor.

## Registro de textos (agregar una fila por archivo)
| Archivo | Carpeta | Título | Autor | Año de muerte del autor | URL de descarga | Quién lo descargó | Qué se quitó a mano |
|---|---|---|---|---|---|---|---|
| (ejemplo) `ejemplo.txt` | train | | | | | | |

## Conteos reales (copiar de `data/clean/stats.json` después de `prepare_data.py`)
| Conjunto | Palabras | Ejemplos | % palabras desconocidas |
|---|---|---|---|
| train | | | |
| val (último 10% de cada libro de train) | | | |
| test | | | |

Notas para el reporte:
- El vocabulario son las 3,000 palabras más frecuentes del entrenamiento; el resto cuenta como `<unk>`.
- Las métricas excluyen las posiciones del test donde la palabra siguiente es desconocida; el porcentaje excluido queda en `results/resultados.csv`.
- Si cambian el test después de congelarlo hay que usar `--unfreeze` y reentrenar y reevaluar los 4 tamaños.
