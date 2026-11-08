"""
Entrenamiento y evaluación del clasificador visual (Avance de proyecto 2).

Datos: TrashNet (Yang y Thung, 2016), 2,527 fotografías en 6 clases de material.
Partición estratificada 70 / 15 / 15 (entrenamiento / validación / prueba), semilla fija.

Experimentos de transferencia de aprendizaje:
  A. Extracción de características (red congelada) con EfficientNet-B0 y con ResNet50:
     se extraen los vectores de la última capa convolucional (con 2 vistas aumentadas
     adicionales por imagen de entrenamiento) y se entrena una cabeza lineal (regresión
     logística) cuyo hiperparámetro C se elige en validación.
  B. Ajuste fino (fine-tuning) de EfficientNet-B0: se parte de la cabeza del experimento A
     y se reentrenan los 2 últimos bloques convolucionales con aumento de datos.

El modelo final se elige por F1 macro en VALIDACIÓN; el conjunto de prueba se usa solo
para reportar. Salidas: modelos/clasificador_residuos.pt y evidencias/avance2/imagen_*.
"""

import argparse
import copy
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from clasificador_imagen import (CLASES, NOMBRE_ES, TRANSF_AUMENTO, TRANSF_EVAL, congelar_hasta, crear_red)

SEMILLA = 2602
RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "evidencias" / "avance2"
MODELOS = RAIZ / "modelos"
URL_TRASHNET = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
DISP = "cuda" if torch.cuda.is_available() else "cpu"


def fijar_semillas(s=SEMILLA):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def obtener_datos(carpeta: Path) -> Path:
    """Devuelve la carpeta con las 6 subcarpetas de TrashNet; la descarga si no existe."""
    destino = carpeta / "dataset-resized"
    if not destino.exists():
        carpeta.mkdir(parents=True, exist_ok=True)
        zip_path = carpeta / "trashnet.zip"
        log("Descargando TrashNet (43 MB)…")
        urllib.request.urlretrieve(URL_TRASHNET, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(carpeta)
    return destino


class Imagenes(Dataset):
    def __init__(self, rutas, etiquetas, transf):
        self.rutas, self.etiquetas, self.transf = list(rutas), list(etiquetas), transf

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        return self.transf(Image.open(self.rutas[i]).convert("RGB")), self.etiquetas[i]


def cargador(rutas, y, transf, mezclar=False, lote=32):
    g = torch.Generator().manual_seed(SEMILLA)
    return DataLoader(Imagenes(rutas, y, transf), batch_size=lote, shuffle=mezclar, generator=g,
                      num_workers=2 if DISP == "cuda" else 0)


def metricas(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": round(accuracy_score(y, p), 4), "precision_macro": round(pr, 4),
            "recall_macro": round(rc, 4), "f1_macro": round(f1, 4)}


@torch.no_grad()
def extraer(red, rutas, transf, vistas=1):
    red.eval()
    salida = []
    for v in range(vistas):
        torch.manual_seed(SEMILLA + v)
        feats = []
        for x, _ in cargador(rutas, [0] * len(rutas), transf):
            feats.append(red.forward_head(red.forward_features(x.to(DISP)), pre_logits=True).cpu())
        salida.append(torch.cat(feats).numpy())
    return salida


@torch.no_grad()
def probabilidades(red, rutas):
    red.eval()
    ps = []
    for x, _ in cargador(rutas, [0] * len(rutas), TRANSF_EVAL, lote=64):
        ps.append(torch.softmax(red(x.to(DISP)), 1).cpu())
    return torch.cat(ps).numpy()


def experimento_a(arq, part, cache: Path):
    """Red congelada + cabeza lineal. Devuelve la red completa con la cabeza ya puesta."""
    fijar_semillas()
    red = crear_red(arq).to(DISP)
    archivo = cache / f"feats_{arq}.npz"
    t0 = time.time()
    if archivo.exists():
        d = np.load(archivo)
        Xtr, Xva, Xte = d["tr"], d["va"], d["te"]
    else:
        log(f"[A:{arq}] extrayendo características…")
        orig = extraer(red, part["tr"][0], TRANSF_EVAL)[0]
        aum = extraer(red, part["tr"][0], TRANSF_AUMENTO, vistas=2)
        Xtr = np.concatenate([orig] + aum)
        Xva = extraer(red, part["va"][0], TRANSF_EVAL)[0]
        Xte = extraer(red, part["te"][0], TRANSF_EVAL)[0]
        np.savez_compressed(archivo, tr=Xtr, va=Xva, te=Xte)
    ytr = np.tile(part["tr"][1], len(Xtr) // len(part["tr"][1]))
    mejor = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        lr = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        f1 = metricas(part["va"][1], lr.predict(Xva))["f1_macro"]
        log(f"[A:{arq}] C={C}  F1 val={f1}")
        if mejor is None or f1 > mejor[0]:
            mejor = (f1, C, lr)
    _, C, lr = mejor
    cabeza = red.get_classifier()
    cabeza.weight.data = torch.tensor(lr.coef_, dtype=torch.float32, device=DISP)
    cabeza.bias.data = torch.tensor(lr.intercept_, dtype=torch.float32, device=DISP)
    seg = round(time.time() - t0, 1)
    return red, {"C": C, "dim_caracteristicas": int(Xtr.shape[1]), "muestras_entrenamiento": int(len(Xtr)),
                 "segundos": seg}


def experimento_b(red_inicial, part, epocas, bloques=2):
    """Ajuste fino de los últimos bloques de EfficientNet-B0 con aumento de datos."""
    fijar_semillas()
    red = copy.deepcopy(red_inicial)
    congelar_hasta(red, "efficientnet_b0", bloques)
    cabeza = list(red.classifier.parameters())
    resto = [p for n, p in red.named_parameters() if p.requires_grad and not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": resto, "lr": 3e-4}, {"params": cabeza, "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    conteo = np.bincount(part["tr"][1], minlength=len(CLASES))
    pesos = torch.tensor((conteo.max() / conteo) ** 0.5, dtype=torch.float32, device=DISP)
    criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.1)
    dl = cargador(*part["tr"], TRANSF_AUMENTO, mezclar=True)
    historia, mejor = [], (-1, None, 0)
    entrenables = sum(p.numel() for p in red.parameters() if p.requires_grad)
    t0 = time.time()
    for ep in range(1, epocas + 1):
        red.train()
        for m in red.modules():   # las BatchNorm congeladas conservan sus estadísticas de ImageNet
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        perdida, n, aciertos = 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(DISP), y.to(DISP)
            opt.zero_grad()
            out = red(x)
            loss = criterio(out, y)
            loss.backward()
            opt.step()
            perdida += loss.item() * len(y); n += len(y); aciertos += (out.argmax(1) == y).sum().item()
        sched.step()
        pva = probabilidades(red, part["va"][0]).argmax(1)
        m = metricas(part["va"][1], pva)
        historia.append({"epoca": ep, "perdida_entrenamiento": round(perdida / n, 4),
                         "accuracy_entrenamiento": round(aciertos / n, 4),
                         "accuracy_validacion": m["accuracy"], "f1_validacion": m["f1_macro"]})
        log(f"[B] época {ep}/{epocas}", historia[-1])
        if m["f1_macro"] > mejor[0]:
            mejor = (m["f1_macro"], copy.deepcopy(red.state_dict()), ep)
    red.load_state_dict(mejor[1])
    return red, historia, {"mejor_epoca": mejor[2], "parametros_entrenables": int(entrenables),
                           "parametros_totales": int(sum(p.numel() for p in red.parameters())),
                           "segundos": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(RAIZ / "datos_imagenes"))
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--cache", default=str(RAIZ / ".cache_caracteristicas"))
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True); MODELOS.mkdir(exist_ok=True)
    cache = Path(args.cache); cache.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    log("Dispositivo:", DISP)

    # ---- Datos y partición ------------------------------------------------------
    carpeta = obtener_datos(Path(args.datos))
    rutas, y = [], []
    for i, c in enumerate(CLASES):
        for f in sorted((carpeta / c).glob("*.jpg")):
            rutas.append(str(f)); y.append(i)
    rutas, y = np.array(rutas), np.array(y)
    r_tr, r_tmp, y_tr, y_tmp = train_test_split(rutas, y, test_size=0.30, stratify=y, random_state=SEMILLA)
    r_va, r_te, y_va, y_te = train_test_split(r_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEMILLA)
    part = {"tr": (r_tr, y_tr), "va": (r_va, y_va), "te": (r_te, y_te)}
    pd.DataFrame([{"archivo": Path(r).name, "clase": CLASES[c], "conjunto": n}
                  for n, (rs, ys) in (("entrenamiento", part["tr"]), ("validacion", part["va"]), ("prueba", part["te"]))
                  for r, c in zip(rs, ys)]).to_csv(SALIDA / "imagen_particion.csv", index=False)
    tabla = pd.DataFrame({n: np.bincount(part[k][1], minlength=6) for k, n in
                          (("tr", "entrenamiento"), ("va", "validacion"), ("te", "prueba"))}, index=CLASES)
    tabla["total"] = tabla.sum(axis=1)
    tabla.to_csv(SALIDA / "imagen_conteos.csv")
    log("Partición:\n" + tabla.to_string())
    with Image.open(rutas[0]) as im:
        resolucion = im.size

    # ---- Experimentos -----------------------------------------------------------
    resultados, modelos = {}, {}
    for arq in ("efficientnet_b0", "resnet50"):
        red, info = experimento_a(arq, part, cache)
        p_va, p_te = probabilidades(red, r_va), probabilidades(red, r_te)
        resultados[f"A_{arq}"] = {"descripcion": f"{arq} congelada + cabeza lineal", **info,
                                  "validacion": metricas(y_va, p_va.argmax(1)), "prueba": metricas(y_te, p_te.argmax(1))}
        modelos[f"A_{arq}"] = (red, arq, p_va, p_te)
        log(f"[A:{arq}]", resultados[f"A_{arq}"])
    red_b, historia, info = experimento_b(modelos["A_efficientnet_b0"][0], part, args.epocas)
    pd.DataFrame(historia).to_csv(SALIDA / "imagen_historia_finetuning.csv", index=False)
    p_va, p_te = probabilidades(red_b, r_va), probabilidades(red_b, r_te)
    resultados["B_efficientnet_b0"] = {"descripcion": "efficientnet_b0 con ajuste fino de 2 bloques", **info,
                                       "validacion": metricas(y_va, p_va.argmax(1)), "prueba": metricas(y_te, p_te.argmax(1))}
    modelos["B_efficientnet_b0"] = (red_b, "efficientnet_b0", p_va, p_te)
    log("[B]", resultados["B_efficientnet_b0"])

    # ---- Selección por validación y evaluación final en prueba -------------------
    elegido = max(resultados, key=lambda k: resultados[k]["validacion"]["f1_macro"])
    red, arq, p_va, p_te = modelos[elegido]
    log("Modelo elegido por F1 en validación:", elegido)

    # Umbral de confianza (validación): por debajo, el bot pide confirmación al usuario.
    filas_u = []
    for u in np.round(np.arange(0.3, 0.96, 0.05), 2):
        m = p_va.max(1) >= u
        filas_u.append({"umbral": float(u), "cobertura": round(m.mean(), 4),
                        "accuracy_cubierta": round(accuracy_score(y_va[m], p_va[m].argmax(1)), 4) if m.any() else None})
    tu = pd.DataFrame(filas_u)
    tu.to_csv(SALIDA / "imagen_umbral.csv", index=False)
    ok = tu[tu["accuracy_cubierta"] >= 0.95]   # al menos 95 % de aciertos en lo que responde sin preguntar
    umbral = float(ok["umbral"].min()) if len(ok) else 0.9

    pred = p_te.argmax(1)
    rep = classification_report(y_te, pred, target_names=CLASES, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.round(4).to_csv(SALIDA / "imagen_reporte_por_clase.csv")
    cm = confusion_matrix(y_te, pred)
    pd.DataFrame(cm, index=CLASES, columns=CLASES).to_csv(SALIDA / "imagen_matriz_confusion.csv")
    pd.DataFrame({"archivo": [Path(r).name for r in r_te], "real": [CLASES[i] for i in y_te],
                  "prediccion": [CLASES[i] for i in pred], "confianza": p_te.max(1).round(4)}
                 ).to_csv(SALIDA / "imagen_predicciones_prueba.csv", index=False)
    m_te = p_te.max(1) >= umbral
    torch.save({"arquitectura": arq, "clases": CLASES, "estado": red.state_dict(), "umbral": umbral,
                "experimento": elegido}, MODELOS / "clasificador_residuos.pt")
    resumen = {"dispositivo": DISP, "imagenes": int(len(rutas)), "resolucion_original": list(resolucion),
               "clases": {c: NOMBRE_ES[c] for c in CLASES}, "particion": tabla.to_dict(),
               "experimentos": resultados, "modelo_final": elegido, "umbral_confianza": umbral,
               "prueba_final": metricas(y_te, pred),
               "prueba_sobre_umbral": {"cobertura": round(m_te.mean(), 4),
                                       "accuracy": round(accuracy_score(y_te[m_te], pred[m_te]), 4)}}
    (SALIDA / "imagen_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    log("Prueba final:", resumen["prueba_final"], "| umbral", umbral, resumen["prueba_sobre_umbral"])


if __name__ == "__main__":
    main()
