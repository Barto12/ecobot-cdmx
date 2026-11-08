"""
Clasificador de intenciones de EcoBot CDMX (Avance de proyecto 2).

Pipeline:  texto -> preprocesamiento NLTK -> representación (TF-IDF o Bag of Words)
                 -> clasificador (Naive Bayes multinomial o SVM lineal) -> intención + confianza

El preprocesamiento reutiliza el del Avance 1 (ecobot.preprocesar) y expone cada etapa
por separado para poder documentarla y hacer ablación:
    1. Normalización: minúsculas, sin acentos (conservando ñ), sin signos ni símbolos.
    2. Tokenización: nltk.word_tokenize (modelo punkt, español).
    3. Eliminación de stopwords: subconjunto de la lista en español de NLTK formado solo por
       artículos, preposiciones, conjunciones y pronombres átonos. La lista completa de NLTK
       borra palabras que en una consulta corta expresan la intención («nos vemos», «muy
       amable», «¿dónde va esto?», «¿hay que lavar?»); la decisión se tomó con un análisis de
       ablación en validación cruzada sobre el entrenamiento (ver entrenar_texto.py).
    4. Stemming: SnowballStemmer('spanish') de NLTK; las interrogativas no se truncan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ecobot import CONSERVAR, STEMMER, STOPWORDS, quitar_acentos

INTENCION_FUERA = "fuera_de_alcance"

# Categorías gramaticales de la lista de NLTK que no aportan a la intención.
SIN_VALOR_DE_INTENCION = {
    "el", "la", "los", "las", "lo", "un", "una", "unos", "unas",            # artículos
    "a", "al", "de", "del", "en", "con", "e", "o", "y", "ni",               # preposiciones y conjunciones
    "se", "le", "les", "su", "sus",                                          # pronombres átonos y posesivos
}
STOPWORDS_NLTK = {quitar_acentos(w) for w in STOPWORDS}
STOPWORDS_ADAPTADA = STOPWORDS_NLTK & SIN_VALOR_DE_INTENCION


# --------------------------------------------------------------------------- #
# Preprocesamiento por etapas
# --------------------------------------------------------------------------- #
def normalizar(texto: str) -> str:
    texto = quitar_acentos(texto.lower())
    texto = re.sub(r"\d+", " ", texto)            # los números no aportan a la intención
    return re.sub(r"[^a-z\s]", " ", texto)


def tokenizar(texto: str) -> list[str]:
    return word_tokenize(texto, language="spanish")


def quitar_stopwords(tokens: list[str], lista: set[str] = STOPWORDS_ADAPTADA) -> list[str]:
    return [t for t in tokens if t not in lista and len(t) > 1]


def stemizar(tokens: list[str]) -> list[str]:
    return [t if t in CONSERVAR else STEMMER.stem(t) for t in tokens]


def etapas(texto: str) -> dict:
    """Devuelve el resultado de cada etapa; se usa para documentar el proceso."""
    n = normalizar(texto)
    t = tokenizar(n)
    s = quitar_stopwords(t)
    return {"original": texto, "normalizado": n.strip(), "tokens": t, "sin_stopwords": s, "raices": stemizar(s)}


@dataclass(frozen=True)
class Preprocesador:
    """Callable configurable: permite apagar etapas para el análisis de ablación.
    Es una clase (y no una lambda) para que el pipeline se pueda guardar con joblib."""
    normalizacion: bool = True
    stopwords: str = "adaptada"        # "adaptada" | "nltk" | "ninguna"
    stemming: bool = True

    def __call__(self, texto: str) -> list[str]:
        texto = normalizar(texto) if self.normalizacion else texto.lower()
        tokens = tokenizar(texto)
        if self.stopwords == "adaptada":
            tokens = quitar_stopwords(tokens, STOPWORDS_ADAPTADA)
        elif self.stopwords == "nltk":
            tokens = quitar_stopwords(tokens, STOPWORDS_NLTK)
        if self.stemming:
            tokens = stemizar(tokens)
        return tokens


# --------------------------------------------------------------------------- #
# Construcción de modelos
# --------------------------------------------------------------------------- #
def vectorizador(tipo: str, prep: Preprocesador | None = None, ngramas=(1, 2)):
    prep = prep or Preprocesador()
    comunes = dict(tokenizer=prep, lowercase=False, token_pattern=None, ngram_range=ngramas)
    if tipo == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **comunes)
    if tipo == "bow":
        return CountVectorizer(**comunes)
    raise ValueError(tipo)


def modelo(tipo: str, **params):
    if tipo == "nb":
        return MultinomialNB(alpha=params.get("alpha", 0.3))
    if tipo == "svm":
        return LinearSVC(C=params.get("C", 1.0), class_weight="balanced", random_state=0)
    raise ValueError(tipo)


def construir(rep: str, clf: str, prep: Preprocesador | None = None, **params) -> Pipeline:
    return Pipeline([("rep", vectorizador(rep, prep)), ("clf", modelo(clf, **params))])


# --------------------------------------------------------------------------- #
# Clasificador listo para el chatbot
# --------------------------------------------------------------------------- #
class ClasificadorIntenciones:
    """Envuelve el pipeline final, calibra probabilidades (para tener una confianza
    comparable entre modelos) y aplica el umbral de fuera de alcance."""

    def __init__(self, rep="tfidf", clf="svm", umbral=0.35, **params):
        self.rep, self.clf, self.umbral, self.params = rep, clf, umbral, params
        self.pipeline = None

    def entrenar(self, textos, etiquetas):
        base = construir(self.rep, self.clf, **self.params)
        if self.clf == "svm":
            # LinearSVC no da probabilidades: se calibran con validación cruzada interna.
            base.steps[-1] = ("clf", CalibratedClassifierCV(base.steps[-1][1], cv=5, method="sigmoid"))
        self.pipeline = base.fit(list(textos), list(etiquetas))
        self.clases = list(self.pipeline.classes_)
        return self

    def probabilidades(self, textos) -> np.ndarray:
        return self.pipeline.predict_proba(list(textos))

    def predecir(self, texto: str) -> tuple[str, float, list[tuple[str, float]]]:
        """Devuelve (intención, confianza, top-3). Si la confianza no supera el
        umbral, la intención se reporta como fuera de alcance."""
        p = self.probabilidades([texto])[0]
        orden = np.argsort(p)[::-1]
        top = [(self.clases[i], float(p[i])) for i in orden[:3]]
        intencion, conf = top[0]
        if conf < self.umbral:
            intencion = INTENCION_FUERA
        return intencion, conf, top

    def guardar(self, ruta):
        joblib.dump({"pipeline": self.pipeline, "rep": self.rep, "clf": self.clf,
                     "umbral": self.umbral, "params": self.params}, ruta)

    @classmethod
    def cargar(cls, ruta):
        d = joblib.load(ruta)
        obj = cls(d["rep"], d["clf"], d["umbral"], **d["params"])
        obj.pipeline = d["pipeline"]
        obj.clases = list(obj.pipeline.classes_)
        return obj
