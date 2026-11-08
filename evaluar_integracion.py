"""
Evaluación de la integración texto-imagen de EcoBot CDMX (Avance de proyecto 2).

1. Intención final del sistema con solo texto, en los dos conjuntos de prueba:
   reglas del Avance 1  vs  clasificador solo  vs  sistema híbrido (clasificador + entidades + política).
2. Prueba de extremo a extremo con imagen: cada una de las 380 fotos de PRUEBA se envía al bot
   con el texto «¿dónde tiro esto?». Se mide si la categoría NADF-024 que comunica el bot es la
   correcta y cuántas veces pide confirmación por confianza baja.
3. Conversaciones de demostración (texto + imagen) guardadas como evidencia.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from clasificador_imagen import A_RESIDUO, CLASES
from ecobot import Contexto
from ecobot_multimodal import EcoBotMultimodal

RAIZ = Path(__file__).resolve().parent
EV = RAIZ / "evidencias" / "avance2"


def reiniciar(bot):
    bot.contexto, bot.candidatos = Contexto(), []


def categoria_real(clase, kb):
    rid = A_RESIDUO[clase]
    return "no_reciclable" if rid is None else next(r["categoria"] for r in kb["residuos"] if r["id"] == rid)


def main(carpeta_imagenes: Path):
    bot = EcoBotMultimodal()
    salida = {}

    # ---- 1. Texto: reglas vs clasificador vs híbrido ----------------------------
    filas = []
    for nombre in ("prueba_externa", "prueba_nueva"):
        df = pd.read_csv(EV / f"texto_predicciones_{nombre}.csv")
        hib = []
        for t in df["texto"]:
            reiniciar(bot)
            bot.responder(t)
            hib.append(bot.turnos[-1].intencion)
        df["hibrido"] = hib
        df.to_csv(EV / f"integracion_texto_{nombre}.csv", index=False)
        filas.append({"conjunto": nombre, "consultas": len(df),
                      "reglas_avance1": round(accuracy_score(df["intencion"], df["reglas_avance1"]), 4),
                      "clasificador": round(accuracy_score(df["intencion"], df["prediccion"]), 4),
                      "hibrido": round(accuracy_score(df["intencion"], df["hibrido"]), 4)})
        salida[f"errores_hibrido_{nombre}"] = df.loc[df["hibrido"] != df["intencion"],
                                                      ["texto", "intencion", "hibrido"]].to_dict("records")
    pd.DataFrame(filas).to_csv(EV / "integracion_texto_resumen.csv", index=False)
    salida["texto"] = filas
    print(pd.DataFrame(filas).to_string(index=False))

    # ---- 2. Extremo a extremo con las fotos de prueba ----------------------------
    part = pd.read_csv(EV / "imagen_particion.csv")
    prueba = part[part["conjunto"] == "prueba"]
    filas = []
    for _, r in prueba.iterrows():
        reiniciar(bot)
        bot.responder("¿dónde tiro esto?", carpeta_imagenes / r["clase"] / r["archivo"])
        t = bot.turnos[-1]
        pide = t.regla.startswith("foto con confianza baja")
        filas.append({"archivo": r["archivo"], "clase_real": r["clase"], "clase_predicha": t.clase_imagen,
                      "confianza": t.confianza_imagen, "pide_confirmacion": pide,
                      "categoria_real": categoria_real(r["clase"], bot.kb),
                      "categoria_comunicada": None if pide else t.categoria,
                      "top1_correcto_si_pide": (t.clase_imagen == r["clase"]) if pide else None,
                      "en_top3_si_pide": (r["clase"] in [c for c, _ in bot.candidatos]) if pide else None})
    e2e = pd.DataFrame(filas)
    e2e.to_csv(EV / "integracion_imagenes_prueba.csv", index=False)
    resp = e2e[~e2e["pide_confirmacion"]]
    dud = e2e[e2e["pide_confirmacion"]]
    salida["extremo_a_extremo"] = {
        "fotos": len(e2e),
        "material_correcto": round((e2e["clase_real"] == e2e["clase_predicha"]).mean(), 4),
        "categoria_correcta_si_se_clasifica_todo": round(
            (e2e["categoria_real"] == e2e["clase_predicha"].map(lambda c: categoria_real(c, bot.kb))).mean(), 4),
        "respondidas_directo": len(resp),
        "categoria_correcta_en_respondidas": round((resp["categoria_real"] == resp["categoria_comunicada"]).mean(), 4),
        "material_correcto_en_respondidas": round((resp["clase_real"] == resp["clase_predicha"]).mean(), 4),
        "pide_confirmacion": len(dud),
        "en_dudosas_top1_correcto": int(dud["top1_correcto_si_pide"].sum()) if len(dud) else 0,
        "en_dudosas_real_en_top3": int(dud["en_top3_si_pide"].sum()) if len(dud) else 0,
    }
    print(json.dumps(salida["extremo_a_extremo"], ensure_ascii=False, indent=2))

    # ---- 3. Conversaciones de demostración ----------------------------------------
    pred = pd.read_csv(EV / "imagen_predicciones_prueba.csv")
    seguras = pred[(pred["real"] == pred["prediccion"]) & (pred["confianza"] > 0.9)]
    def foto(clase, i=0):
        return carpeta_imagenes / clase / seguras[seguras["real"] == clase].iloc[i]["archivo"]
    dudosa = pred[pred["confianza"] < bot.vision.umbral].sort_values("confianza").iloc[len(pred[pred["confianza"] < bot.vision.umbral]) // 2]
    escenarios = [
        ("D1 · Foto + pregunta de clasificación", [("¿dónde tiro esto?", foto("glass"))]),
        ("D2 · Foto + pregunta de días (la intención del texto se aplica al material de la foto)",
         [("¿qué día pasan por esto?", foto("cardboard"))]),
        ("D3 · Solo foto, sin texto", [("", foto("plastic"))]),
        ("D4 · Texto que condiciona la respuesta visual (regla de contaminación)",
         [("está lleno de grasa, ¿se recicla?", foto("cardboard", 1))]),
        ("D5 · Foto dudosa: el bot pide confirmación y el usuario responde",
         [("¿en qué bote va?", carpeta_imagenes / dudosa["real"] / dudosa["archivo"]), (f"es {dict(cardboard='cartón', glass='vidrio', metal='metal', paper='papel', plastic='plástico', trash='basura')[dudosa['real']]}", None)]),
        ("D6 · Residuo que no está en la base: el bot pide foto y la usa",
         [("¿dónde tiro este envase?", None), ("", foto("metal"))]),
        ("D7 · Clase de aprovechamiento limitado", [("¿esto se recicla?", foto("trash"))]),
        ("D8 · Conversación de texto con contexto (híbrido)",
         [("hola", None), ("¿y el periódico?", None), ("¿qué día lo recogen?", None), ("gracias", None)]),
    ]
    lineas, demo = ["EcoBot CDMX multimodal — conversaciones de demostración", "=" * 72], []
    for titulo, turnos in escenarios:
        reiniciar(bot)
        lineas += ["", titulo, "-" * 72]
        for texto, img in turnos:
            resp = bot.responder(texto, img)
            t = bot.turnos[-1]
            lineas.append(f"Usuario: {texto or '(sin texto)'}" + (f"   [imagen: {Path(img).name}]" if img else ""))
            meta = f"  ↳ intención={t.intencion} (conf. {t.confianza_texto:.2f})"
            if t.clase_imagen:
                meta += f" | imagen={t.clase_imagen} (p={t.confianza_imagen:.2f})"
            lineas += [meta + f" | regla={t.regla}", "EcoBot: " + resp.replace("\n", "\n        ")]
            demo.append({"escenario": titulo, "texto": texto, "imagen": str(img) if img else "",
                         "intencion": t.intencion, "clase_imagen": t.clase_imagen,
                         "confianza_imagen": t.confianza_imagen, "regla": t.regla, "respuesta": resp})
    (EV / "integracion_conversaciones.txt").write_text("\n".join(lineas), encoding="utf-8")
    pd.DataFrame(demo).to_csv(EV / "integracion_conversaciones.csv", index=False)
    (EV / "integracion_resumen.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "datos_imagenes" / "dataset-resized")
