# EcoBot CDMX

Asistente inteligente que orienta a residentes de la Ciudad de México sobre la separación de residuos conforme a la Norma Ambiental **NADF-024-AMBT-2013**: en qué contenedor va cada residuo, qué días se recoge, cómo prepararlo y a dónde llevar los residuos de manejo especial. Desde el Avance 2 entiende texto libre con un clasificador de intenciones y reconoce el material de una **foto** con una red neuronal convolucional.

Proyecto por fases de *Desarrollo de Aplicaciones con Inteligencia Artificial* (MBIA2602) · Maestría en Inteligencia Artificial, Universidad Tecmilenio.
Autor: Nazir Enoch Rosas Salgado · Profesor: Mackenson Polche.

## Ejecución en Google Colab

| Avance | Cuaderno | Qué hace |
|---|---|---|
| 1 | `EcoBot_CDMX.ipynb` | Chatbot basado en reglas con NLTK y sus pruebas |
| 2 | `EcoBot_CDMX_Avance2.ipynb` | Clasificador de intenciones, CNN con transferencia de aprendizaje e integración texto-imagen |

Para el Avance 2: abre el cuaderno en Colab, elige `Entorno de ejecución → Cambiar tipo de entorno → GPU`, y luego `Ejecutar todas`. El cuaderno clona este repositorio, descarga TrashNet y los pesos preentrenados, entrena y evalúa todo (≈5 min con GPU).

## Ejecución local

```bash
pip install -r requirements.txt
python datos/generar_dataset_texto.py      # conjunto de datos textual
python entrenar_texto.py                   # TF-IDF/BoW × Naive Bayes/SVM, ablación, evaluación
python entrenar_imagen.py --epocas 8       # EfficientNet-B0 / ResNet50 con transferencia (descarga TrashNet)
python evaluar_integracion.py              # integración texto-imagen y conversaciones de demostración
python ecobot_multimodal.py                # chat interactivo; para enviar foto: foto:ruta.jpg ¿dónde va?
```

## Estructura

| Archivo | Contenido |
|---|---|
| `base_conocimiento.json` | Categorías NADF-024, 33 residuos con alias, programas de SEDEMA (v1.1) |
| `intenciones.json` | Intenciones y respuestas del chatbot |
| `ecobot.py` | Motor del Avance 1: preprocesamiento NLTK, entidades, política de diálogo |
| `datos/generar_dataset_texto.py` | Construye el conjunto de entrenamiento (675 frases, 13 intenciones) y los de prueba |
| `clasificador_texto.py` | Preprocesamiento por etapas, TF-IDF / Bag of Words, Naive Bayes / SVM, umbral de confianza |
| `entrenar_texto.py` | Selección por validación cruzada, ablación del preprocesamiento, evaluación en prueba |
| `clasificador_imagen.py` | CNN (EfficientNet-B0 o ResNet50) con transferencia de aprendizaje desde ImageNet |
| `entrenar_imagen.py` | Partición 70/15/15, extracción de características, ajuste fino, métricas |
| `ecobot_multimodal.py` | Bot integrado: texto + imagen → base de conocimiento → respuesta |
| `evaluar_integracion.py` | Evaluación del sistema híbrido y prueba de extremo a extremo con fotos |
| `evidencias/` | Avance 1 (reglas) y `evidencias/avance2/` (métricas, matrices, predicciones, conversaciones) |
| `modelos/` | `intenciones.joblib` (clasificador de texto). La CNN se regenera con `entrenar_imagen.py`. |

## Datos y modelos de terceros

- **TrashNet** (Yang y Thung, 2016): 2,527 fotos en 6 materiales, descargado de [garythung/trashnet](https://github.com/garythung/trashnet).
- Pesos ImageNet de **EfficientNet-B0** y **ResNet50** publicados por la biblioteca [timm](https://github.com/huggingface/pytorch-image-models).

## Aviso

Los días de recolección pueden variar por alcaldía y ruta. Las fechas de programas corresponden al calendario 2026 de SEDEMA y deben confirmarse antes de acudir.
