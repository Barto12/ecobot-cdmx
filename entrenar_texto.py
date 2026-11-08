"""
Entrenamiento y evaluación del clasificador de intenciones (Avance de proyecto 2).

Protocolo:
  1. Selección de modelo SOLO con el conjunto de entrenamiento: validación cruzada
     estratificada de 5 particiones sobre 4 combinaciones (TF-IDF / BoW x Naive Bayes / SVM)
     con búsqueda de hiperparámetros. Métrica de selección: F1 macro.
  2. Ablación del preprocesamiento sobre la mejor combinación.
  3. Umbral de confianza elegido con predicciones fuera de partición (out-of-fold) del entrenamiento.
  4. Evaluación final, una sola vez, sobre dos conjuntos que no participaron en el entrenamiento:
     prueba_externa (57 consultas del Avance 1) y prueba_nueva (52 consultas escritas antes de entrenar).
  5. Comparación con el reconocimiento por reglas del Avance 1 sobre los mismos conjuntos.

Salidas: modelos/intenciones.joblib y evidencias/avance2/texto_*.{csv,json}
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from clasificador_texto import (INTENCION_FUERA, ClasificadorIntenciones, Preprocesador, construir, etapas)
from ecobot import RAIZ, EcoBot

SEMILLA = 2602
DATOS = RAIZ / "datos"
SALIDA = RAIZ / "evidencias" / "avance2"
MODELOS = RAIZ / "modelos"
SALIDA.mkdir(parents=True, exist_ok=True)
MODELOS.mkdir(exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
REJILLA = {"nb": {"clf__alpha": [0.05, 0.1, 0.3, 1.0]}, "svm": {"clf__C": [0.1, 0.3, 1.0, 3.0]}}
NOMBRE = {"tfidf": "TF-IDF", "bow": "Bag of Words", "nb": "Naive Bayes", "svm": "SVM lineal"}


def metricas(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": round(accuracy_score(y, p), 4), "precision_macro": round(pr, 4),
            "recall_macro": round(rc, 4), "f1_macro": round(f1, 4)}


def main():
    t0 = time.time()
    entr = pd.read_csv(DATOS / "intenciones_entrenamiento.csv")
    ext = pd.read_csv(DATOS / "prueba_externa.csv")
    nva = pd.read_csv(DATOS / "prueba_nueva.csv")
    X, y = entr["texto"].tolist(), entr["intencion"].tolist()
    resumen = {"entrenamiento": len(entr), "prueba_externa": len(ext), "prueba_nueva": len(nva),
               "intenciones": sorted(set(y)), "frases_por_intencion": entr["intencion"].value_counts().to_dict(),
               "origen": entr["origen"].value_counts().to_dict()}

    # ---- 1. Ejemplos del preprocesamiento y tamaño de vocabulario ----------------
    ejemplos = ["¿Dónde tiro las botellas de plástico?", "oye, ¿qué días pasan por los orgánicos?",
                "¿Hay que lavar los frascos antes de reciclarlos?", "¿El Reciclatrón recibe celulares?"]
    resumen["ejemplos_preprocesamiento"] = [etapas(e) for e in ejemplos]
    vocab = {}
    for nombre, prep in [("sin_preprocesamiento", Preprocesador(False, "ninguna", False)),
                         ("normalizacion", Preprocesador(True, "ninguna", False)),
                         ("normalizacion_stopwords", Preprocesador(True, "adaptada", False)),
                         ("completo", Preprocesador(True, "adaptada", True))]:
        vocab[nombre] = len({tok for t in X for tok in prep(t)})
    resumen["vocabulario_unigramas"] = vocab
    print("Vocabulario:", vocab)

    # ---- 2. Comparación de modelos con validación cruzada ------------------------
    filas = []
    mejores = {}
    for rep in ("tfidf", "bow"):
        for clf in ("nb", "svm"):
            gs = GridSearchCV(construir(rep, clf), REJILLA[clf], cv=CV, scoring="f1_macro", n_jobs=1)
            gs.fit(X, y)
            mejor = gs.best_estimator_
            cvr = cross_validate(mejor, X, y, cv=CV, scoring=["accuracy", "f1_macro"])
            pe, pn = mejor.predict(ext["texto"]), mejor.predict(nva["texto"])
            fila = {"representacion": NOMBRE[rep], "clasificador": NOMBRE[clf],
                    "hiperparametro": json.dumps({k.split("__")[1]: v for k, v in gs.best_params_.items()}),
                    "cv_accuracy": round(cvr["test_accuracy"].mean(), 4),
                    "cv_f1_macro": round(cvr["test_f1_macro"].mean(), 4),
                    "cv_f1_macro_std": round(cvr["test_f1_macro"].std(), 4),
                    "externa_accuracy": round(accuracy_score(ext["intencion"], pe), 4),
                    "nueva_accuracy": round(accuracy_score(nva["intencion"], pn), 4),
                    "nueva_f1_macro": metricas(nva["intencion"], pn)["f1_macro"]}
            filas.append(fila)
            mejores[(rep, clf)] = gs.best_params_
            print(fila)
    comp = pd.DataFrame(filas)
    comp.to_csv(SALIDA / "texto_comparacion_modelos.csv", index=False)
    idx = comp["cv_f1_macro"].idxmax()
    rep_m = "tfidf" if comp.loc[idx, "representacion"] == "TF-IDF" else "bow"
    clf_m = "nb" if comp.loc[idx, "clasificador"] == "Naive Bayes" else "svm"
    params = {k.split("__")[1]: v for k, v in mejores[(rep_m, clf_m)].items()}
    resumen["modelo_seleccionado"] = {"representacion": NOMBRE[rep_m], "clasificador": NOMBRE[clf_m], **params}
    print("Seleccionado por CV:", resumen["modelo_seleccionado"])

    # ---- 3. Ablación del preprocesamiento ---------------------------------------
    abl = []
    for nombre, prep in [("Completo, stopwords adaptadas (final)", Preprocesador(True, "adaptada", True)),
                         ("Completo, stopwords NLTK sin adaptar", Preprocesador(True, "nltk", True)),
                         ("Sin eliminar stopwords", Preprocesador(True, "ninguna", True)),
                         ("Sin stemming", Preprocesador(True, "adaptada", False)),
                         ("Sin normalización", Preprocesador(False, "adaptada", True)),
                         ("Solo tokenización", Preprocesador(False, "ninguna", False))]:
        pipe = construir(rep_m, clf_m, prep, **params)
        cvr = cross_validate(pipe, X, y, cv=CV, scoring=["accuracy", "f1_macro"])
        pipe.fit(X, y)
        abl.append({"configuracion": nombre, "cv_f1_macro": round(cvr["test_f1_macro"].mean(), 4),
                    "nueva_accuracy": round(accuracy_score(nva["intencion"], pipe.predict(nva["texto"])), 4),
                    "externa_accuracy": round(accuracy_score(ext["intencion"], pipe.predict(ext["texto"])), 4)})
        print(abl[-1])
    pd.DataFrame(abl).to_csv(SALIDA / "texto_ablacion.csv", index=False)

    # ---- 4. Umbral de confianza con predicciones out-of-fold ----------------------
    final = ClasificadorIntenciones(rep_m, clf_m, umbral=0.0, **params)
    base = construir(rep_m, clf_m, **params)
    if clf_m == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        base.steps[-1] = ("clf", CalibratedClassifierCV(base.steps[-1][1], cv=5, method="sigmoid"))
    oof = cross_val_predict(base, X, y, cv=CV, method="predict_proba")
    clases = sorted(set(y))
    umbrales = []
    for u in np.round(np.arange(0.0, 0.61, 0.05), 2):
        pred = [clases[i] if p[i] >= u else INTENCION_FUERA for p, i in zip(oof, oof.argmax(1))]
        umbrales.append({"umbral": float(u), **metricas(y, pred)})
    tabla_u = pd.DataFrame(umbrales)
    tabla_u.to_csv(SALIDA / "texto_umbral.csv", index=False)
    mejor_u = float(tabla_u.loc[tabla_u["f1_macro"].idxmax(), "umbral"])
    final.umbral = mejor_u
    resumen["umbral_confianza"] = mejor_u
    print("Umbral elegido:", mejor_u)

    # ---- 5. Modelo final y evaluación en los conjuntos de prueba ------------------
    final.entrenar(X, y)
    final.guardar(MODELOS / "intenciones.joblib")
    reglas = EcoBot()
    resultados = {}
    todas = []
    for nombre, df in (("prueba_externa", ext), ("prueba_nueva", nva)):
        preds, confs, reg = [], [], []
        for t in df["texto"]:
            i, c, _ = final.predecir(t)
            preds.append(i)
            confs.append(round(c, 3))
            bot = EcoBot()          # reglas del Avance 1, un turno sin contexto
            bot.responder(t)
            r = bot.historial[-1].intencion
            reg.append(INTENCION_FUERA if r == "fallback" else r)
        df = df.assign(prediccion=preds, confianza=confs, reglas_avance1=reg)
        df["correcto"] = df["prediccion"] == df["intencion"]
        df.to_csv(SALIDA / f"texto_predicciones_{nombre}.csv", index=False)
        todas.append(df)
        resultados[nombre] = {"clasificador": metricas(df["intencion"], df["prediccion"]),
                              "reglas_avance1": metricas(df["intencion"], df["reglas_avance1"]),
                              "errores": df.loc[~df["correcto"], ["texto", "intencion", "prediccion", "confianza"]]
                                           .to_dict("records")}
        print(nombre, resultados[nombre]["clasificador"], "| reglas:", resultados[nombre]["reglas_avance1"])
    resumen["resultados"] = resultados

    conj = pd.concat(todas)
    etiquetas = sorted(set(conj["intencion"]) | set(conj["prediccion"]))
    rep = classification_report(conj["intencion"], conj["prediccion"], labels=etiquetas,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.round(3).to_csv(SALIDA / "texto_reporte_por_clase.csv")
    cm = confusion_matrix(conj["intencion"], conj["prediccion"], labels=etiquetas)
    pd.DataFrame(cm, index=etiquetas, columns=etiquetas).to_csv(SALIDA / "texto_matriz_confusion.csv")
    resumen["pruebas_combinadas"] = metricas(conj["intencion"], conj["prediccion"])
    resumen["segundos"] = round(time.time() - t0, 1)
    (SALIDA / "texto_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Combinado:", resumen["pruebas_combinadas"], f"({resumen['segundos']} s)")


if __name__ == "__main__":
    main()
