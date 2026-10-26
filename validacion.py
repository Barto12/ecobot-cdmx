"""
Set de validación independiente: 20 consultas de un solo turno que NO se usaron
para ajustar las reglas. Se ejecuta una vez para estimar el desempeño fuera de
las conversaciones de prueba.
"""

import csv
import json

from ecobot import EcoBot, RAIZ

CASOS = [
    ("¿En qué bote van los huesos de pollo?", "clasificar_residuo", ["huesos"]),
    ("¿Dónde pongo el cartón de huevo?", "clasificar_residuo", ["cartón"]),
    ("las servilletas usadas a dónde van", "clasificar_residuo", []),
    ("¿Qué día recogen la basura inorgánica?", "dias_recoleccion", []),
    ("¿El vidrio roto se recicla?", "clasificar_residuo", ["vidrio"]),
    ("tengo una laptop vieja", "clasificar_residuo", ["electrónicos"]),
    ("¿Las colillas de cigarro en qué contenedor?", "clasificar_residuo", ["colillas"]),
    ("¿Qué hago con un refri que ya no sirve?", "clasificar_residuo", ["voluminosos"]),
    ("¿Puedo tirar el aceite por el fregadero?", "clasificar_residuo", ["aceite de cocina usado"]),
    ("¿A qué hora pasa el camión?", "dias_recoleccion", []),
    ("¿Dónde queda el próximo mercado de trueque?", "mercado_trueque", []),
    ("buen día, tengo una duda", "saludo", []),
    ("¿Qué contenedor es el naranja?", "colores", []),
    ("¿Las bolsas de papitas se reciclan?", "clasificar_residuo", ["empaques metalizados"]),
    ("¿Hay multa si no separo?", "normativa", []),
    ("¿Dónde tiro el cubrebocas?", "clasificar_residuo", []),
    ("¿Las cajas de leche son reciclables?", "clasificar_residuo", ["tetrapak"]),
    ("¿Qué onda con el reciclatrón, qué reciben?", "reciclatron", []),
    ("restos de jardín, ¿dónde van?", "clasificar_residuo", ["residuos de jardinería"]),
    ("¿Recomiéndame una película?", "fallback", []),
]

if __name__ == "__main__":
    filas = []
    for entrada, esperada, ents in CASOS:
        bot = EcoBot()
        resp = bot.responder(entrada)
        t = bot.historial[-1]
        filas.append({"entrada": entrada, "intencion_esperada": esperada, "intencion_obtenida": t.intencion,
                      "entidades_esperadas": "|".join(ents), "entidades_obtenidas": "|".join(t.entidades),
                      "ok": t.intencion == esperada and sorted(t.entidades) == sorted(ents),
                      "respuesta": resp})
    ok = sum(f["ok"] for f in filas)
    with open(RAIZ / "evidencias" / "validacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(("OK   " if f["ok"] else "FALLA"), f["entrada"], "→", f["intencion_obtenida"], f["entidades_obtenidas"] or "")
        if not f["ok"]:
            print("      ", f["respuesta"][:160].replace("\n", " "))
    print(json.dumps({"casos": len(filas), "correctos": ok, "exactitud": round(ok / len(filas), 3)}, ensure_ascii=False))
