# EcoBot CDMX

Chatbot basado en reglas que orienta a residentes de la Ciudad de México sobre la separación de residuos conforme a la Norma Ambiental **NADF-024-AMBT-2013**: en qué contenedor va cada residuo, qué días se recoge, cómo prepararlo y a dónde llevar los residuos de manejo especial.

Proyecto Fase 1 · *Desarrollo de Aplicaciones con Inteligencia Artificial* (MBIA2602) · Maestría en Inteligencia Artificial, Universidad Tecmilenio.
Autor: Nazir Enoch Rosas Salgado · Profesor: Mackenson Polche.

## Ejecución en Google Colab

1. Abre `EcoBot_CDMX.ipynb` en Colab (Archivo → Subir cuaderno, o desde GitHub).
2. `Entorno de ejecución → Ejecutar todas`.
3. La última celda abre la conversación interactiva; escribe `salir` para terminar.

## Ejecución local

```bash
pip install -r requirements.txt
python ecobot.py        # conversación interactiva
python pruebas.py       # 12 conversaciones de prueba → evidencias/
python validacion.py    # 20 consultas de validación independiente
```

Requiere Python 3.10 o superior.

## Estructura

| Archivo | Contenido |
|---|---|
| `base_conocimiento.json` | Categorías NADF-024, 33 residuos con alias, programas Reciclatrón y Mercado de Trueque |
| `intenciones.json` | 12 intenciones con patrones y respuestas, más respuestas de respaldo |
| `ecobot.py` | Motor: preprocesamiento NLTK, detección de residuos, puntuación de intenciones, política de diálogo con contexto |
| `pruebas.py` | 12 conversaciones (37 turnos) con intención y residuos esperados |
| `validacion.py` | 20 consultas no usadas para ajustar reglas |
| `EcoBot_CDMX.ipynb` | Notebook autocontenido para Colab |
| `evidencias/` | Transcripciones, resultados por turno (CSV) y resumen de métricas |

## Uso de NLTK

- `word_tokenize` (punkt, español) para tokenizar.
- `stopwords.words('spanish')` para filtrar palabras vacías, conservando interrogativas y la negación.
- `SnowballStemmer('spanish')` para reducir variantes («botella», «botellas» → `botell`).
- `nltk.util.ngrams` para detectar residuos de varias palabras («botella de vidrio»).
- `nltk.metrics.edit_distance` para tolerar errores de una letra («botela de bidrio»).

## Resultados

| Conjunto | Turnos | Correctos | Exactitud |
|---|---|---|---|
| Conversaciones de prueba (v0.1, primera corrida) | 37 | 31 | 83.8 % |
| Conversaciones de prueba (v1.0, tras correcciones) | 37 | 37 | 100 % |
| Validación independiente (v1.0) | 20 | 17 | 85 % |

La cifra de 100 % se obtuvo sobre las mismas conversaciones usadas para corregir el motor; la estimación honesta del desempeño es la de validación.

## Aviso

Los días de recolección pueden variar por alcaldía y ruta. La información de programas corresponde al calendario 2026 de SEDEMA y debe confirmarse antes de acudir.
