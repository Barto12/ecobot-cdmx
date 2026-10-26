"""
Pruebas funcionales de EcoBot CDMX: 12 conversaciones distintas.

Cada turno declara la intención esperada y, cuando aplica, los residuos esperados.
El script ejecuta cada conversación con un bot nuevo (contexto limpio), compara lo
obtenido contra lo esperado y guarda evidencias en la carpeta evidencias/.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from ecobot import EcoBot, RAIZ

CONVERSACIONES = [
    {"id": "C01", "perfil": "Residente que separa por primera vez",
     "turnos": [("Hola", "saludo", []),
                ("¿Dónde tiro las cáscaras de plátano?", "clasificar_residuo", ["cáscaras"]),
                ("¿Y qué día pasan?", "dias_recoleccion", []),
                ("Gracias", "agradecimiento", [])]},
    {"id": "C02", "perfil": "Usuario con electrónicos y pilas",
     "turnos": [("Buenas tardes", "saludo", []),
                ("Tengo pilas viejas y un celular que ya no sirve, ¿qué hago con ellos?", "clasificar_residuo", ["pilas", "electrónicos"]),
                ("¿Cuándo es el próximo Reciclatrón?", "reciclatron", []),
                ("Adiós", "despedida", [])]},
    {"id": "C03", "perfil": "Duda sobre papel y cartón contaminados",
     "turnos": [("¿La caja de pizza con grasa es reciclable?", "clasificar_residuo", ["cartón"]),
                ("¿Y el periódico?", "clasificar_residuo", ["papel"]),
                ("¿De qué color es el contenedor de reciclables?", "colores", [])]},
    {"id": "C04", "perfil": "Consulta de calendario y categorías",
     "turnos": [("¿Qué días pasa el camión de la basura?", "dias_recoleccion", []),
                ("¿Qué va en los inorgánicos no reciclables?", "info_categoria", []),
                ("¿Los pañales?", "clasificar_residuo", ["pañales"])]},
    {"id": "C05", "perfil": "Aceite de cocina y preparación de envases",
     "turnos": [("¿Dónde tiro el aceite de cocina usado?", "clasificar_residuo", ["aceite de cocina usado"]),
                ("¿Qué es el Mercado de Trueque?", "mercado_trueque", []),
                ("¿Hay que lavar las botellas antes de tirarlas?", "preparacion", ["botellas de PET"])]},
    {"id": "C06", "perfil": "Usuario que escribe con errores ortográficos",
     "turnos": [("donde va la botela de bidrio", "clasificar_residuo", ["vidrio"]),
                ("y las latas de atun", "clasificar_residuo", ["latas y metales"]),
                ("ok grax", "agradecimiento", [])]},
    {"id": "C07", "perfil": "Consulta incompleta que requiere aclaración",
     "turnos": [("Quiero tirar algo", "clasificar_residuo", []),
                ("unas pilas", "clasificar_residuo", ["pilas"]),
                ("Gracias", "agradecimiento", [])]},
    {"id": "C08", "perfil": "Pregunta sobre la normativa",
     "turnos": [("¿Es obligatorio separar la basura?", "normativa", []),
                ("¿Por qué separar si al final la revuelven?", "normativa", []),
                ("Hasta luego", "despedida", [])]},
    {"id": "C09", "perfil": "Preguntas fuera de alcance",
     "turnos": [("¿Cómo va a estar el clima mañana?", "fallback", []),
                ("¿Quién ganó el partido de ayer?", "fallback", []),
                ("ayuda", "ayuda", [])]},
    {"id": "C10", "perfil": "Residuos voluminosos",
     "turnos": [("Tengo un colchón viejo, ¿cómo me deshago de él?", "clasificar_residuo", ["voluminosos"]),
                ("¿Y qué día pasan por los voluminosos?", "dias_recoleccion", [])]},
    {"id": "C11", "perfil": "Varios residuos en una sola pregunta",
     "turnos": [("¿Dónde van las medicinas caducas y los focos ahorradores?", "clasificar_residuo", ["medicamentos caducos", "focos ahorradores y lámparas"]),
                ("¿El Reciclatrón recibe focos?", "reciclatron", ["focos ahorradores y lámparas"]),
                ("Perfecto, gracias", "agradecimiento", [])]},
    {"id": "C12", "perfil": "Residuo no incluido en la base de conocimiento",
     "turnos": [("¿Dónde tiro una bolsa de plástico?", "clasificar_residuo", []),
                ("¿Y los residuos orgánicos qué incluyen?", "info_categoria", []),
                ("¿Se recicla el unicel?", "clasificar_residuo", ["unicel"])]},
]


def ejecutar() -> dict:
    carpeta = RAIZ / "evidencias"
    carpeta.mkdir(exist_ok=True)
    filas, transcripcion = [], []
    sello = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    transcripcion.append(f"EcoBot CDMX — Evidencias de ejecución ({sello})\n" + "=" * 72)

    for conv in CONVERSACIONES:
        bot = EcoBot()  # contexto limpio por conversación
        transcripcion.append(f"\n[{conv['id']}] {conv['perfil']}\n" + "-" * 72)
        for n, (entrada, esperada, ents_esp) in enumerate(conv["turnos"], 1):
            respuesta = bot.responder(entrada)
            t = bot.historial[-1]
            ok_int = t.intencion == esperada
            ok_ent = sorted(t.entidades) == sorted(ents_esp)
            filas.append({
                "conversacion": conv["id"], "turno": n, "entrada": entrada,
                "tokens": " ".join(t.tokens), "intencion_esperada": esperada,
                "intencion_obtenida": t.intencion, "entidades_esperadas": "|".join(ents_esp),
                "entidades_obtenidas": "|".join(t.entidades), "categoria": t.categoria or "",
                "regla": t.regla, "intencion_ok": ok_int, "entidad_ok": ok_ent,
                "respuesta": respuesta,
            })
            marca = "OK" if ok_int and ok_ent else "FALLA"
            transcripcion.append(f"Usuario: {entrada}")
            transcripcion.append(f"  ↳ tokens={t.tokens} | intención={t.intencion} | entidades={t.entidades} | regla={t.regla} [{marca}]")
            transcripcion.append("EcoBot: " + respuesta.replace("\n", "\n        ") + "\n")

    total = len(filas)
    resumen = {
        "fecha": sello,
        "conversaciones": len(CONVERSACIONES),
        "turnos": total,
        "intenciones_correctas": sum(f["intencion_ok"] for f in filas),
        "entidades_correctas": sum(f["entidad_ok"] for f in filas),
        "turnos_correctos": sum(f["intencion_ok"] and f["entidad_ok"] for f in filas),
        "conversaciones_sin_falla": sum(
            all(f["intencion_ok"] and f["entidad_ok"] for f in filas if f["conversacion"] == c["id"])
            for c in CONVERSACIONES),
    }
    resumen["exactitud_intencion"] = round(resumen["intenciones_correctas"] / total, 3)
    resumen["exactitud_turno"] = round(resumen["turnos_correctos"] / total, 3)

    with open(carpeta / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    (carpeta / "conversaciones.txt").write_text("\n".join(transcripcion), encoding="utf-8")
    (carpeta / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"resumen": resumen, "filas": filas, "transcripcion": "\n".join(transcripcion)}


if __name__ == "__main__":
    r = ejecutar()
    print(r["transcripcion"])
    print("\nResumen:", json.dumps(r["resumen"], ensure_ascii=False, indent=2))
