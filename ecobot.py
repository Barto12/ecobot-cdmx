"""
EcoBot CDMX — Chatbot basado en reglas para orientar la separación de residuos
en la Ciudad de México conforme a la Norma Ambiental NADF-024-AMBT-2013.

Proyecto Fase 1 · Desarrollo de Aplicaciones con Inteligencia Artificial (MBIA2602)
Autor: Nazir Enoch Rosas Salgado · Universidad Tecmilenio

Pipeline por turno:
    entrada -> normalización -> tokenización (NLTK) -> filtrado de stopwords (NLTK)
            -> stemming Snowball (NLTK) -> extracción de entidades (residuo / categoría)
            -> puntuación de intenciones por reglas -> política de diálogo con contexto
            -> respuesta
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import nltk
from nltk.metrics.distance import edit_distance
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

RAIZ = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# --------------------------------------------------------------------------- #
# Recursos NLTK
# --------------------------------------------------------------------------- #
def asegurar_recursos_nltk() -> None:
    """Descarga los recursos de NLTK que el bot necesita, solo si faltan."""
    for recurso, ruta in [("stopwords", "corpora/stopwords"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(recurso, quiet=True)


asegurar_recursos_nltk()
from nltk.corpus import stopwords  # noqa: E402  (se importa después de descargar)

# Palabras interrogativas y la negación se conservan: cambian la intención
# («¿dónde?», «¿cuándo?») o la categoría («no reciclable»).
CONSERVAR = {"no", "donde", "cuando", "que", "como", "cual"}
STOPWORDS = {w for w in stopwords.words("spanish")} - CONSERVAR
STEMMER = SnowballStemmer("spanish")


# --------------------------------------------------------------------------- #
# Preprocesamiento
# --------------------------------------------------------------------------- #
def quitar_acentos(texto: str) -> str:
    # La ñ se conserva aparte para no convertir «año» en «ano».
    texto = texto.replace("ñ", "\0")
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return texto.replace("\0", "n")


def preprocesar(texto: str) -> list[str]:
    """Normaliza, tokeniza con NLTK, quita stopwords y aplica stemming."""
    texto = quitar_acentos(texto.lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    tokens = word_tokenize(texto, language="spanish")
    # Las interrogativas no se truncan: Snowball reduce «cómo» y «comida» a la misma raíz «com».
    return [t if t in CONSERVAR else STEMMER.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


def similares(a: str, b: str, prefijo: bool = True, min_edicion: int = 5) -> bool:
    """Compara dos raíces tolerando errores de una letra (distancia de edición de NLTK)
    y, solo para entidades, variaciones por prefijo («celul» / «celular»)."""
    if a == b:
        return True
    corto = min(len(a), len(b))
    if prefijo and corto >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    return corto >= min_edicion and edit_distance(a, b) <= 1


def peso(raiz: str) -> float:
    """Las interrogativas aportan menos evidencia que las palabras de contenido."""
    return 0.5 if raiz in CONSERVAR else 1.0


# --------------------------------------------------------------------------- #
# Estructuras
# --------------------------------------------------------------------------- #
@dataclass
class Contexto:
    """Memoria de corto plazo de la conversación."""
    ultima_categoria: str | None = None
    ultimo_residuo: str | None = None
    esperando_residuo: bool = False
    fallos_seguidos: int = 0


@dataclass
class Turno:
    entrada: str
    tokens: list[str]
    intencion: str
    puntaje: float
    entidades: list[str]
    categoria: str | None
    respuesta: str
    regla: str = ""


@dataclass
class EcoBot:
    ruta_kb: Path = RAIZ / "base_conocimiento.json"
    ruta_intenciones: Path = RAIZ / "intenciones.json"
    contexto: Contexto = field(default_factory=Contexto)
    historial: list[Turno] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kb = json.loads(Path(self.ruta_kb).read_text(encoding="utf-8"))
        datos = json.loads(Path(self.ruta_intenciones).read_text(encoding="utf-8"))
        self.intenciones = {i["tag"]: i for i in datos["intenciones"]}
        self.fallback = datos["fallback"]
        self._rotacion: dict[str, int] = {}

        # Patrones de intención preprocesados con el mismo pipeline que la entrada.
        self.patrones = {
            tag: [preprocesar(p) for p in i["patrones"]] for tag, i in self.intenciones.items()
        }
        # Alias de residuos preprocesados, ordenados del más largo al más corto
        # para que «botella de vidrio» gane sobre «botella».
        self.alias = []
        for r in self.kb["residuos"]:
            for a in r["alias"]:
                raices = preprocesar(a)
                if raices:
                    self.alias.append((raices, r))
        self.alias.sort(key=lambda x: -len(x[0]))
        self.contaminantes = {STEMMER.stem(quitar_acentos(m)) for m in self.kb["modificadores_contaminacion"]}

    # ------------------------------------------------------------------ #
    # Reconocimiento
    # ------------------------------------------------------------------ #
    def detectar_residuos(self, tokens: list[str]) -> list[dict]:
        """Busca residuos de la base de conocimiento con n-gramas (NLTK).
        Primera pasada: coincidencia exacta. Segunda: coincidencia aproximada."""
        usados: set[int] = set()
        hallados: list[dict] = []
        for exacto in (True, False):
            for raices, residuo in self.alias:
                n = len(raices)
                for i, grama in enumerate(ngrams(tokens, n)):
                    pos = set(range(i, i + n))
                    if pos & usados:
                        continue
                    # Un alias de una sola palabra exige raíces más largas para tolerar
                    # errores; así «mañana» (manan) no se confunde con «manzana» (manzan).
                    min_ed = 6 if n == 1 else 5
                    ok = all((g == r) if exacto else similares(g, r, min_edicion=min_ed)
                             for g, r in zip(grama, raices))
                    if ok:
                        usados |= pos
                        if residuo not in hallados:
                            hallados.append(residuo)
        return hallados

    @staticmethod
    def detectar_categoria(tokens: list[str]) -> str | None:
        t = " ".join(tokens)
        if re.search(r"\bno recicl", t):
            return "no_reciclable"
        if re.search(r"\brecicl(?!atron)", t):
            return "reciclable"
        if re.search(r"\borgan", t):
            return "organico"
        if re.search(r"\b(especial|volumin)", t):
            return "manejo_especial"
        return None

    def puntuar_intenciones(self, tokens: list[str]) -> tuple[str | None, float]:
        """Cada patrón suma el peso de sus raíces si TODAS aparecen en la entrada.
        Gana la intención con mayor puntaje; en empate, la que aparece primero."""
        conjunto = set(tokens)
        mejor, mejor_p = None, 0.0
        for tag, patrones in self.patrones.items():
            p = sum(sum(peso(x) for x in pt) for pt in patrones
                    if pt and all(any(similares(x, y, prefijo=False) for y in conjunto) for x in pt))
            if p > mejor_p:
                mejor, mejor_p = tag, p
        return mejor, mejor_p

    # ------------------------------------------------------------------ #
    # Respuestas
    # ------------------------------------------------------------------ #
    def _rotar(self, opciones: list[str], clave: str) -> str:
        i = self._rotacion.get(clave, 0)
        self._rotacion[clave] = i + 1
        return opciones[i % len(opciones)]

    def _cat(self, clave: str) -> dict:
        return self.kb["categorias"][clave]

    def respuesta_residuo(self, residuo: dict, tokens: list[str]) -> tuple[str, str]:
        cat_clave = residuo["categoria"]
        # Regla de contaminación: papel o cartón sucio deja de ser reciclable.
        if residuo.get("condicionado") and any(similares(t, m) for t in tokens for m in self.contaminantes):
            cat_clave = "no_reciclable"
            c = self._cat(cat_clave)
            texto = (f"Si el {residuo['id']} está sucio, mojado o con grasa ya no se puede reciclar: "
                     f"va en {c['nombre'].upper()} (contenedor {c['color']}). Se recoge {c['dias']}.")
            return texto, cat_clave
        c = self._cat(cat_clave)
        if cat_clave == "manejo_especial":
            texto = (f"{residuo['id'].capitalize()} es un residuo de MANEJO ESPECIAL (contenedor {c['color']}). "
                     f"No lo entregues al camión ordinario; llévalo a {residuo['destino']}")
        else:
            texto = (f"{residuo['id'].capitalize()} → {c['nombre'].upper()} (contenedor {c['color']}). "
                     f"Se recoge {c['dias']}. {c['preparacion']}")
        if residuo.get("condicionado"):
            texto += " Ojo: solo si está limpio y seco."
        return texto, cat_clave

    def respuesta_dias(self, cat: str | None) -> str:
        if cat:
            c = self._cat(cat)
            if cat == "manejo_especial":
                return f"Los residuos de {c['nombre'].lower()} {c['dias']}. {c['preparacion']}"
            return f"{c['nombre']} se recogen {c['dias']}. {self.kb['metadatos']['aviso']}"
        org, rec = self._cat("organico"), self._cat("reciclable")
        return (f"Calendario general de recolección en CDMX:\n"
                f"  • Orgánicos: {org['dias']}\n"
                f"  • Inorgánicos (reciclables y no reciclables): {rec['dias']}\n"
                f"  • Manejo especial y voluminosos: no van en la recolección ordinaria\n"
                f"{self.kb['metadatos']['aviso']}")

    def respuesta_categoria(self, cat: str) -> str:
        c = self._cat(cat)
        return (f"{c['nombre']} (contenedor {c['color']}): {c['descripcion']}. "
                f"Recolección: {c['dias']}.")

    def respuesta_programa(self, clave: str) -> str:
        p = self.kb["programas"][clave]
        if clave == "reciclatron":
            return (f"{p['nombre']}: {p['descripcion']} No recibe {p['no_recibe']}. "
                    f"Próximas jornadas: {p['proxima']}.")
        return (f"{p['nombre']}: {p['descripcion']} Recibe {p['recibe']}. "
                f"Próximas fechas: {p['proximas']}.")

    # ------------------------------------------------------------------ #
    # Política de diálogo
    # ------------------------------------------------------------------ #
    def responder(self, entrada: str) -> str:
        tokens = preprocesar(entrada)
        residuos = self.detectar_residuos(tokens)
        categoria = self.detectar_categoria(tokens) if not residuos else None
        intencion, puntaje = self.puntuar_intenciones(tokens)
        ctx = self.contexto

        # Si hay un residuo y la intención es débil o de clasificación, se clasifica.
        if residuos and intencion in (None, "clasificar_residuo", "info_categoria", "ayuda", "colores"):
            intencion = "clasificar_residuo"

        respuesta, regla, cat_resp = "", "", None

        if intencion == "dias_recoleccion":
            cat = residuos[0]["categoria"] if residuos else (categoria or ctx.ultima_categoria)
            respuesta = self.respuesta_dias(cat)
            regla = "días con categoría" if cat else "días generales"
            cat_resp = cat

        elif intencion == "clasificar_residuo" and residuos:
            partes = []
            for r in residuos:
                txt, cat_resp = self.respuesta_residuo(r, tokens)
                partes.append(txt)
            respuesta = "\n".join(partes)
            regla = "clasificación por base de conocimiento"
            ctx.ultimo_residuo = residuos[-1]["id"]

        elif intencion == "clasificar_residuo":
            respuesta = "¿Qué residuo quieres tirar? Dime el objeto, por ejemplo «cáscaras», «pilas» o «botella de vidrio»."
            regla = "falta entidad: se pide aclaración"
            ctx.esperando_residuo = True

        elif intencion == "preparacion":
            cat = residuos[0]["categoria"] if residuos else (categoria or ctx.ultima_categoria)
            base = self._rotar(self.intenciones["preparacion"]["respuestas"], "preparacion")
            respuesta = base + (f"\nPara {self._cat(cat)['nombre'].lower()}: {self._cat(cat)['preparacion']}" if cat else "")
            regla = "preparación"
            cat_resp = cat

        elif intencion == "reciclatron":
            respuesta, regla = self.respuesta_programa("reciclatron"), "programa"

        elif intencion == "mercado_trueque":
            respuesta, regla = self.respuesta_programa("mercado_trueque"), "programa"

        elif intencion == "info_categoria" and (categoria or ctx.ultima_categoria):
            cat_resp = categoria or ctx.ultima_categoria
            respuesta, regla = self.respuesta_categoria(cat_resp), "descripción de categoría"

        elif intencion is None and categoria:
            cat_resp = categoria
            respuesta, regla = self.respuesta_categoria(categoria), "categoría mencionada sin verbo"

        elif intencion == "colores" and (categoria or ctx.ultima_categoria):
            cat_resp = categoria or ctx.ultima_categoria
            c = self._cat(cat_resp)
            respuesta = f"El contenedor de {c['nombre'].lower()} es {c['color'].upper()}."
            regla = "color de una categoría"

        elif intencion in ("saludo", "despedida", "agradecimiento", "ayuda", "colores", "normativa"):
            respuesta = self._rotar(self.intenciones[intencion]["respuestas"], intencion)
            regla = "respuesta estática"

        elif residuos:
            # Residuo detectado pero intención no prevista: se prioriza clasificar.
            intencion = "clasificar_residuo"
            txt, cat_resp = self.respuesta_residuo(residuos[0], tokens)
            respuesta, regla = txt, "residuo sin intención explícita"

        else:
            intencion = "fallback"
            ctx.fallos_seguidos += 1
            if ctx.fallos_seguidos >= 2:
                respuesta = self.intenciones["ayuda"]["respuestas"][0]
                regla = "fallback repetido: se muestra menú"
            else:
                respuesta = self._rotar(self.fallback, "fallback")
                regla = "fallback"

        if intencion != "fallback":
            ctx.fallos_seguidos = 0
        if intencion == "clasificar_residuo" and residuos:
            ctx.esperando_residuo = False
        if cat_resp:
            ctx.ultima_categoria = cat_resp

        self.historial.append(Turno(entrada, tokens, intencion or "fallback", puntaje,
                                    [r["id"] for r in residuos], cat_resp, respuesta, regla))
        return respuesta

    def reiniciar(self) -> None:
        self.contexto = Contexto()
        self.historial = []


def conversar() -> None:
    """Modo interactivo para Colab o terminal."""
    bot = EcoBot()
    print("EcoBot CDMX — escribe «salir» para terminar.\n")
    while True:
        entrada = input("Tú: ").strip()
        if not entrada:
            continue
        print("EcoBot:", bot.responder(entrada), "\n")
        if bot.historial[-1].intencion == "despedida":
            break


if __name__ == "__main__":
    conversar()
