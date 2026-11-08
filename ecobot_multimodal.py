"""
EcoBot CDMX multimodal (Avance de proyecto 2): integra el componente conversacional y el visual.

Cada turno puede traer texto, una imagen o ambos, y pasa por tres operaciones:
  1. Procesamiento de texto: el clasificador de intenciones (TF-IDF/BoW + SVM/NB) decide la
     intención; las reglas del Avance 1 extraen los residuos mencionados (entidades).
  2. Clasificación de imagen: la CNN (transferencia de aprendizaje) reconoce el material.
  3. Interpretación y comunicación: la política de diálogo combina ambas señales, traduce
     el material a la categoría de la NADF-024 con la base de conocimiento y explica el
     resultado al usuario, con su nivel de confianza; si la confianza es baja, pide confirmación.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clasificador_imagen import A_RESIDUO, NOMBRE_ES, ClasificadorResiduos
from clasificador_texto import INTENCION_FUERA, ClasificadorIntenciones
from ecobot import RAIZ, EcoBot, preprocesar, quitar_acentos

# Intenciones «débiles» que ceden ante un residuo mencionado o una foto.
DEBILES = {"clasificar_residuo", INTENCION_FUERA, "info_categoria", "ayuda", "colores", "saludo"}
# Intenciones que, si llega una foto, se responden sobre el material de la foto.
SOBRE_LA_FOTO = {"dias_recoleccion", "preparacion", "colores"}
# Índice inverso color -> categoría (mejora planteada en el Avance 1: ir del atributo a la categoría).
COLOR_A_CATEGORIA = {"verd": "organico", "gris": "reciclable", "naranj": "no_reciclable", "caf": "manejo_especial"}
CONTEXTO_COLOR = {"bot", "contenedor", "color", "colores", "bolsa"}
AFIRMACION = re.compile(r"^\s*(si|sip|simon|correcto|exacto|asi es|ajam|ok|va|claro)\b")


@dataclass
class TurnoMultimodal:
    texto: str
    imagen: str | None
    intencion: str
    confianza_texto: float
    entidades: list[str]
    clase_imagen: str | None
    confianza_imagen: float | None
    categoria: str | None
    regla: str
    respuesta: str


@dataclass
class EcoBotMultimodal(EcoBot):
    ruta_texto: str = str(RAIZ / "modelos" / "intenciones.joblib")
    ruta_imagen: str = str(RAIZ / "modelos" / "clasificador_residuos.pt")
    turnos: list = field(default_factory=list)
    candidatos: list = field(default_factory=list)   # top-3 de la última foto dudosa

    def __post_init__(self) -> None:
        super().__post_init__()
        self.nlu = ClasificadorIntenciones.cargar(self.ruta_texto)
        self.vision = ClasificadorResiduos(self.ruta_imagen)
        self.por_id = {r["id"]: r for r in self.kb["residuos"]}

    # ------------------------------------------------------------------ #
    def _respuesta_material(self, clase: str, tokens: list[str]) -> tuple[str, str]:
        """Traduce el material reconocido a la categoría NADF-024 usando la base de conocimiento."""
        rid = A_RESIDUO[clase]
        if rid is None:   # clase «trash»: residuo de aprovechamiento limitado sin objeto específico
            c = self._cat("no_reciclable")
            return (f"Parece un residuo de aprovechamiento limitado → {c['nombre'].upper()} "
                    f"(contenedor {c['color']}). Se recoge {c['dias']}. Si en realidad es un envase "
                    f"limpio de plástico, vidrio, metal o cartón, va con los reciclables."), "no_reciclable"
        return self.respuesta_residuo(self.por_id[rid], tokens)

    @staticmethod
    def _color_mencionado(tokens: list[str]) -> str | None:
        """Devuelve la categoría si el texto nombra un color de contenedor. «Café» solo cuenta
        junto a «bote», «contenedor» o «color», para no confundirlo con el residuo café."""
        for t in tokens:
            for raiz, cat in COLOR_A_CATEGORIA.items():
                if t.startswith(raiz) and (raiz != "caf" or CONTEXTO_COLOR & set(tokens)):
                    return cat
        return None

    def _interpretar_foto(self, imagen) -> tuple[str, float, list]:
        clase, p, top = self.vision.predecir(imagen)
        return clase, p, top

    # ------------------------------------------------------------------ #
    def responder(self, texto: str = "", imagen=None) -> str:
        texto = texto or ""
        tokens = preprocesar(texto) if texto else []
        residuos = self.detectar_residuos(tokens)
        categoria_txt = self.detectar_categoria(tokens) if not residuos else None
        ctx = self.contexto

        # 1) Procesamiento de texto
        if texto:
            intencion, conf_t, _ = self.nlu.predecir(texto)
        else:
            intencion, conf_t = "clasificar_residuo", 1.0     # foto sin texto = «¿dónde va esto?»

        # 2) Clasificación de imagen
        clase = p_img = None
        top = []
        if imagen is not None:
            clase, p_img, top = self._interpretar_foto(imagen)

        respuesta, regla, cat = "", "", None

        # Confirmación de una foto dudosa del turno anterior
        if self.candidatos and imagen is None and not residuos and AFIRMACION.match(quitar_acentos(texto.lower())):
            clase = self.candidatos[0][0]
            texto_kb, cat = self._respuesta_material(clase, tokens)
            respuesta = f"Perfecto, lo tomo como {NOMBRE_ES[clase]}. {texto_kb}"
            regla, intencion = "confirmación de foto dudosa", "clasificar_residuo"
            self.candidatos = []

        # 3a) Hay foto: se interpreta según la intención del texto
        elif clase is not None and (intencion in DEBILES or intencion in SOBRE_LA_FOTO) and not residuos:
            if p_img < self.vision.umbral:
                alternativas = ", ".join(f"{NOMBRE_ES[c]} ({q:.0%})" for c, q in top[1:3])
                respuesta = (f"En la foto creo ver {NOMBRE_ES[clase]} ({p_img:.0%} de confianza), pero no estoy seguro; "
                             f"también podría ser {alternativas}. ¿Me confirmas qué material es? "
                             f"Puedes responder «sí» o escribir el material.")
                regla, self.candidatos = "foto con confianza baja: pide confirmación", top
                intencion = "clasificar_residuo"
            elif intencion in SOBRE_LA_FOTO:
                cat = "no_reciclable" if A_RESIDUO[clase] is None else self.por_id[A_RESIDUO[clase]]["categoria"]
                prefijo = f"En la foto veo {NOMBRE_ES[clase]} ({p_img:.0%})."
                if intencion == "dias_recoleccion":
                    respuesta = f"{prefijo} {self.respuesta_dias(cat)}"
                elif intencion == "preparacion":
                    respuesta = f"{prefijo} {self._cat(cat)['preparacion']}"
                else:
                    respuesta = f"{prefijo} Va en el contenedor {self._cat(cat)['color'].upper()}."
                regla = f"{intencion} sobre el material de la foto"
            else:
                texto_kb, cat = self._respuesta_material(clase, tokens)
                respuesta = f"En la foto veo {NOMBRE_ES[clase]} ({p_img:.0%} de confianza). {texto_kb}"
                regla, intencion = "clasificación por imagen", "clasificar_residuo"
                self.candidatos = []

        # 3b) Pregunta inversa por color («¿qué va en el bote naranja?»)
        elif (cat_color := self._color_mencionado(tokens)) and intencion in DEBILES | {"colores"}:
            cat = cat_color
            c = self._cat(cat)
            respuesta = (f"El contenedor {c['color'].upper()} es para {c['nombre'].lower()}: {c['descripcion']}. "
                         f"Recolección: {c['dias']}.")
            regla, intencion = "color → categoría (índice inverso)", "colores"

        # 3c) Solo texto, con residuo mencionado: prioridad del residuo (regla del Avance 1)
        elif residuos and intencion in DEBILES:
            partes = []
            for r in residuos:
                t, cat = self.respuesta_residuo(r, tokens)
                partes.append(t)
            respuesta, regla, intencion = "\n".join(partes), "residuo mencionado (entidad)", "clasificar_residuo"
            self.candidatos = []

        elif intencion == "clasificar_residuo":
            respuesta = ("No tengo ese residuo en mi base o no lo reconocí. ¿Me mandas una foto o me dices "
                         "de qué material es (plástico, vidrio, papel, cartón, metal…)?")
            regla = "falta el residuo: pide foto o material"

        elif intencion == "dias_recoleccion":
            cat = residuos[0]["categoria"] if residuos else (categoria_txt or ctx.ultima_categoria)
            respuesta, regla = self.respuesta_dias(cat), "días"
        elif intencion == "preparacion":
            cat = residuos[0]["categoria"] if residuos else (categoria_txt or ctx.ultima_categoria)
            base = self.intenciones["preparacion"]["respuestas"][0]
            respuesta = base + (f"\nPara {self._cat(cat)['nombre'].lower()}: {self._cat(cat)['preparacion']}" if cat else "")
            regla = "preparación"
        elif intencion == "reciclatron":
            respuesta, regla = self.respuesta_programa("reciclatron"), "programa"
        elif intencion == "mercado_trueque":
            respuesta, regla = self.respuesta_programa("mercado_trueque"), "programa"
        elif intencion == "info_categoria" and (categoria_txt or ctx.ultima_categoria):
            cat = categoria_txt or ctx.ultima_categoria
            respuesta, regla = self.respuesta_categoria(cat), "descripción de categoría"
        elif intencion == "colores" and (categoria_txt or ctx.ultima_categoria):
            cat = categoria_txt or ctx.ultima_categoria
            c = self._cat(cat)
            respuesta, regla = f"El contenedor de {c['nombre'].lower()} es {c['color'].upper()}.", "color de categoría"
        elif intencion in self.intenciones and self.intenciones[intencion]["respuestas"]:
            respuesta, regla = self._rotar(self.intenciones[intencion]["respuestas"], intencion), "respuesta de la intención"
        else:
            intencion = INTENCION_FUERA
            ctx.fallos_seguidos += 1
            respuesta = (self.intenciones["ayuda"]["respuestas"][0] if ctx.fallos_seguidos >= 2
                         else self._rotar(self.fallback, "fallback"))
            regla = "fuera de alcance"

        if intencion != INTENCION_FUERA:
            ctx.fallos_seguidos = 0
        if cat:
            ctx.ultima_categoria = cat
        self.turnos.append(TurnoMultimodal(texto, str(imagen) if imagen is not None else None, intencion,
                                           round(conf_t, 3), [r["id"] for r in residuos], clase,
                                           None if p_img is None else round(p_img, 3), cat, regla, respuesta))
        return respuesta


def conversar() -> None:
    """Modo interactivo: escribe texto; para mandar una foto escribe  foto:/ruta/imagen.jpg  [texto]."""
    bot = EcoBotMultimodal()
    print("EcoBot CDMX multimodal — «foto:ruta.jpg ¿dónde va?» para enviar imagen, «salir» para terminar.\n")
    while True:
        entrada = input("Tú: ").strip()
        if not entrada:
            continue
        imagen = None
        if entrada.startswith("foto:"):
            partes = entrada[5:].split(maxsplit=1)
            imagen, entrada = partes[0], (partes[1] if len(partes) > 1 else "")
        print("EcoBot:", bot.responder(entrada, imagen), "\n")
        if bot.turnos[-1].intencion == "despedida":
            break


if __name__ == "__main__":
    conversar()
