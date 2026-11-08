"""
Construcción del conjunto de datos textual para entrenar el clasificador de intenciones
de EcoBot CDMX (Avance de proyecto 2).

Fuentes de las frases:
  * manual    : frases escritas a mano por intención, en español coloquial de México.
  * plantilla : combinaciones de plantillas de pregunta con residuos, categorías y prefijos.
  * ruido     : copias de frases manuales con un error de tecleo simulado (solo entrenamiento).

Salidas (carpeta datos/):
  intenciones_entrenamiento.csv  -> texto, intencion, origen
  prueba_externa.csv             -> 57 consultas de las pruebas del Avance 1 (no se usan para entrenar)
  prueba_nueva.csv               -> 52 consultas nuevas escritas antes de entrenar (4 por intención)

Se fija la semilla para que el conjunto sea reproducible.
"""

import csv
import random
import re
import sys
import unicodedata
from pathlib import Path

SEMILLA = 2602
random.seed(SEMILLA)
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

# --------------------------------------------------------------------------- #
# Vocabulario de apoyo
# --------------------------------------------------------------------------- #
# (texto con artículo, es_plural). Incluye residuos de la base de conocimiento y
# residuos que NO están en ella, para que el modelo aprenda la intención por el
# verbo y la estructura, no por el nombre del objeto.
RESIDUOS = [
    ("las cáscaras de naranja", True), ("los restos de comida", True), ("los huesos de pollo", True),
    ("el café molido usado", False), ("las hojas secas", True), ("los cascarones de huevo", True),
    ("las tortillas duras", True), ("el pasto cortado", False),
    ("las botellas de plástico", True), ("el garrafón", False), ("las latas de refresco", True),
    ("los frascos de vidrio", True), ("el cartón", False), ("las cajas de cereal", True),
    ("el periódico viejo", False), ("los cuadernos usados", True), ("el tetrapak", False),
    ("la ropa vieja", False), ("los envases de yogurt", True), ("las tapas de plástico", True),
    ("los pañales", True), ("el papel de baño", False), ("el unicel", False),
    ("las colillas", True), ("los chicles", True), ("las bolsas de frituras", True),
    ("los zapatos rotos", True), ("las plumas que ya no pintan", True), ("los cotonetes", True),
    ("las pilas", True), ("el celular viejo", False), ("la computadora descompuesta", False),
    ("el aceite de cocina", False), ("las medicinas caducadas", True), ("los focos ahorradores", True),
    ("el colchón", False), ("el sillón viejo", False), ("las llantas", True), ("el escombro", False),
    ("los cables", True), ("el cargador", False),
    # fuera de la base de conocimiento
    ("el cubrebocas", False), ("las servilletas usadas", True), ("la bolsa de plástico", False),
    ("el cepillo de dientes", False), ("las cápsulas de café", True), ("la esponja de la cocina", False),
    ("los vasos desechables", True), ("el rastrillo", False), ("las jeringas", True),
    ("el popote", False), ("los platos de cartón sucios", True), ("las cajas de pizza", True),
]
CATEGORIAS = ["los orgánicos", "la basura orgánica", "los inorgánicos", "la basura inorgánica",
              "los reciclables", "lo reciclable", "los no reciclables", "la basura no reciclable",
              "los residuos de manejo especial", "los voluminosos"]
CATEGORIAS_SIN_ART = ["orgánicos", "inorgánicos", "reciclables", "no reciclables", "manejo especial",
                      "voluminosos", "basura orgánica", "residuos inorgánicos"]
COLORES = ["verde", "gris", "naranja", "café"]
DIAS = ["el lunes", "el martes", "el miércoles", "el jueves", "el viernes", "el sábado", "el domingo",
        "hoy", "mañana", "los fines de semana"]
PREFIJOS = ["", "", "", "oye, ", "una pregunta, ", "disculpa, ", "oye bot, ", "hola, ", "perdón, ", "oigan, "]


def conj(plural: bool, sg: str, pl: str) -> str:
    return pl if plural else sg


# --------------------------------------------------------------------------- #
# Frases por intención
# --------------------------------------------------------------------------- #
def clasificar_residuo():
    plantillas = [
        lambda r, p: f"¿dónde tiro {r}?",
        lambda r, p: f"¿en qué bote {conj(p, 'va', 'van')} {r}?",
        lambda r, p: f"¿{r} {conj(p, 'es reciclable', 'son reciclables')}?",
        lambda r, p: f"¿{conj(p, 'se recicla', 'se reciclan')} {r}?",
        lambda r, p: f"¿qué hago con {r}?",
        lambda r, p: f"¿{r} {conj(p, 'va', 'van')} en orgánicos o inorgánicos?",
        lambda r, p: f"¿a qué contenedor {conj(p, 'va', 'van')} {r}?",
        lambda r, p: f"tengo {r}, ¿dónde {conj(p, 'lo', 'los')} pongo?",
        lambda r, p: f"¿cómo separo {r}?",
        lambda r, p: f"{r}, ¿dónde {conj(p, 'va', 'van')}?",
        lambda r, p: f"¿{r} en cuál bote?",
        lambda r, p: f"no sé dónde echar {r}",
        lambda r, p: f"¿dónde deposito {r}?",
        lambda r, p: f"¿{r} {conj(p, 'se tira', 'se tiran')} con lo reciclable?",
        lambda r, p: f"¿en qué bolsa pongo {r}?",
        lambda r, p: f"¿cómo me deshago de {r}?",
        lambda r, p: f"¿{r} {conj(p, 'es', 'son')} basura orgánica?",
        lambda r, p: f"¿a dónde {conj(p, 'va', 'van')} a parar {r}?",
        lambda r, p: f"¿puedo tirar {r} a la basura normal?",
        lambda r, p: f"¿{r} {conj(p, 'cuenta', 'cuentan')} como reciclable?",
    ]
    frases = []
    for r, p in RESIDUOS:
        for f in random.sample(plantillas, 3):
            frases.append(("plantilla", random.choice(PREFIJOS) + f(r, p)))
    manuales = [
        "¿dónde va esto?", "¿en qué bote va esto?", "¿qué es esto y dónde lo tiro?",
        "te mando una foto, ¿en qué contenedor va?", "mira esta foto, ¿esto se recicla?",
        "¿esto es orgánico o inorgánico?", "quiero tirar algo pero no sé dónde",
        "¿esto lo puedo reciclar?", "¿a qué bote le toca esto?", "¿y esto dónde lo echo?",
        "ayúdame a separar esta basura", "¿cómo clasifico este residuo?",
        "¿esto va con la basura reciclable?", "subí una imagen, dime dónde va",
        "no sé si esto es reciclable", "¿en cuál contenedor pongo esta cosa?",
        "¿qué tipo de residuo es este?", "¿esto es basura de manejo especial?",
        "¿lo tiro en el verde o en el gris?", "tengo varias cosas para tirar, ¿me ayudas a separarlas?",
    ]
    frases += [("manual", f) for f in manuales]
    return frases


def dias_recoleccion():
    frases = []
    for c in CATEGORIAS:
        for f in random.sample([
            f"¿qué día pasan por {c}?", f"¿cuándo recogen {c}?", f"¿qué días se llevan {c}?",
            f"¿cuándo pasa el camión de {c}?", f"¿qué días toca sacar {c}?",
        ], 2):
            frases.append(("plantilla", random.choice(PREFIJOS) + f))
    for d in DIAS:
        f = random.choice([f"¿{d} pasa el camión?", f"¿{d} recogen basura?", f"¿qué basura recogen {d}?",
                           f"¿{d} toca orgánica o inorgánica?"])
        frases.append(("plantilla", f))
    for r, p in random.sample(RESIDUOS, 8):
        frases.append(("plantilla", f"¿qué día {conj(p, 'recogen', 'recogen')} {r}?"))
    manuales = [
        "¿qué días pasa el camión?", "¿a qué hora pasa la basura?", "¿cada cuándo pasa el camión?",
        "¿qué días sacan la basura en mi colonia?", "calendario de recolección", "horario del camión de basura",
        "¿cuándo pasan?", "¿qué días recogen?", "¿cuándo puedo sacar la basura?",
        "¿todos los días pasa el camión?", "¿el camión pasa diario?", "¿qué día me toca sacar la basura?",
        "¿cuándo es la recolección?", "¿qué días hay recolección diferenciada?",
        "¿el camión de la basura pasa en la mañana o en la tarde?", "¿en qué días se recoge cada tipo de basura?",
        "dame los días de recolección", "¿cuál es el calendario de la basura en 2026?",
    ]
    frases += [("manual", f) for f in manuales]
    return frases


def info_categoria():
    frases = []
    for c in CATEGORIAS_SIN_ART:
        for f in random.sample([
            f"¿qué va en {c}?", f"¿qué son los {c}?" if not c.startswith("basura") else f"¿qué es la {c}?",
            f"¿qué entra en {c}?", f"dame ejemplos de {c}", f"¿qué se considera {c}?",
            f"¿qué incluye la categoría de {c}?", f"explícame qué es {c}",
        ], 3):
            frases.append(("plantilla", random.choice(PREFIJOS) + f))
    manuales = [
        "¿cuáles son las categorías de basura?", "¿en cuántos grupos se separa la basura?",
        "¿cómo se clasifica la basura en la ciudad?", "¿qué tipos de residuos hay?",
        "¿cuál es la diferencia entre reciclable y no reciclable?",
        "¿qué diferencia hay entre orgánico e inorgánico?", "explícame las cuatro categorías",
        "¿qué significa manejo especial?", "¿qué cosas son de aprovechamiento limitado?",
        "¿qué es un residuo voluminoso?", "¿qué cuenta como basura sanitaria?",
        "¿cuáles son los residuos inorgánicos reciclables?",
    ]
    frases += [("manual", f) for f in manuales]
    return frases


def colores():
    frases = []
    for c in COLORES:
        for f in random.sample([
            f"¿para qué es el bote {c}?", f"¿qué va en el contenedor {c}?", f"¿el {c} qué basura es?",
            f"¿qué se tira en el bote {c}?", f"¿el contenedor {c} es para qué?",
        ], 3):
            frases.append(("plantilla", f))
    for c in CATEGORIAS:
        f = random.choice([f"¿de qué color es el bote de {c.split(' ', 1)[1]}?",
                           f"¿qué color le toca a {c}?", f"¿en qué color de bote van {c}?"])
        frases.append(("plantilla", f))
    manuales = [
        "código de colores", "¿qué significan los colores de los botes?", "¿qué color es cada bote?",
        "¿de qué colores son los contenedores?", "¿hay un color para cada tipo de basura?",
        "colores de los botes de basura", "¿los botes tienen colores?", "¿qué color uso para la basura orgánica?",
        "¿el gris es para inorgánicos?", "¿de qué color es el contenedor de manejo especial?",
    ]
    frases += [("manual", f) for f in manuales]
    return frases


def preparacion():
    frases = []
    plantillas = [
        lambda r, p: f"¿hay que lavar {r}?", lambda r, p: f"¿tengo que enjuagar {r} antes de tirar{conj(p, 'lo', 'los')}?",
        lambda r, p: f"¿le quito la etiqueta a {r}?", lambda r, p: f"¿aplasto {r}?",
        lambda r, p: f"¿puedo tirar {r} {conj(p, 'sucio', 'sucios')}?", lambda r, p: f"¿cómo preparo {r} para reciclar?",
        lambda r, p: f"¿{r} {conj(p, 'tiene', 'tienen')} que ir {conj(p, 'limpio', 'limpios')}?",
        lambda r, p: f"¿cómo entrego {r}?",
    ]
    reciclables = [x for x in RESIDUOS if any(k in x[0] for k in
                   ("botella", "garrafón", "lata", "frasco", "cartón", "caja", "envase", "tetrapak", "tapa", "periódico"))]
    for r, p in reciclables:
        for f in random.sample(plantillas, 3):
            frases.append(("plantilla", random.choice(PREFIJOS) + f(r, p)))
    manuales = [
        "¿cómo preparo los reciclables?", "¿los envases tienen que ir limpios?", "¿separo las tapas de las botellas?",
        "¿los orgánicos van en bolsa?", "¿escurro los restos de comida?", "¿hay que secar los envases?",
        "¿cómo debo entregar la basura?", "¿tengo que lavar todo antes de tirarlo?",
        "¿cómo empaco la basura para el camión?", "¿junto el papel en una caja o en bolsa?",
        "¿cómo guardo el aceite usado?", "¿cómo envuelvo los residuos sanitarios?",
    ]
    frases += [("manual", f) for f in manuales]
    return frases


def reciclatron():
    manuales = [
        "¿cuándo es el reciclatrón?", "¿dónde hay reciclatrón?", "¿qué reciben en el reciclatrón?",
        "¿dónde llevo mis electrónicos viejos?", "centro de acopio de electrónicos", "¿aceptan televisores en el reciclatrón?",
        "próxima jornada de reciclaje de electrónicos", "¿el reciclatrón recibe pilas?", "¿hay un reciclatrón permanente?",
        "¿a qué hora abre el reciclatrón?", "¿cuándo es el próximo reciclatrón?", "¿qué no reciben en el reciclatrón?",
        "¿dónde se pone el reciclatrón este mes?", "¿el reciclatrón acepta computadoras rotas?", "fechas del reciclatrón",
        "¿qué es el reciclatrón?", "¿hay acopio de aparatos eléctricos?", "¿dónde reciclo aparatos electrónicos?",
        "¿el reciclatrón cobra?", "¿qué necesito para llevar cosas al reciclatrón?", "mega reciclatrón",
        "¿en qué alcaldía es el siguiente reciclatrón?", "¿dónde entrego mi tele vieja para reciclar?",
        "¿hay jornadas de acopio de electrónicos en noviembre?", "¿el reciclatrón recibe focos?",
        "¿puedo llevar un refri al reciclatrón?", "¿dónde queda el reciclatrón de Coyoacán?",
        "¿cuál es el horario del reciclatrón?", "¿qué aparatos se pueden llevar al reciclatrón?",
        "jornada de acopio de residuos electrónicos",
    ]
    frases = [("manual", f) for f in manuales]
    for f in random.sample(manuales, 10):
        frases.append(("plantilla", random.choice(PREFIJOS[3:]) + f))
    return frases


def mercado_trueque():
    manuales = [
        "¿qué es el mercado de trueque?", "¿cuándo es el próximo trueque?", "¿dónde se pone el mercado de trueque?",
        "¿qué puedo cambiar en el trueque?", "¿cómo funcionan los puntos verdes?", "¿cuánto material necesito para el trueque?",
        "quiero cambiar mis reciclables por comida", "¿qué me dan a cambio de mi basura?", "¿el trueque acepta aceite?",
        "intercambio de reciclables por hortalizas", "¿dónde cambio mi basura por plantas?", "fechas del mercado de trueque",
        "¿qué recibe el mercado de trueque?", "¿cuántos puntos verdes me dan por kilo?", "mercado de trueque de diciembre",
        "¿el mercado de trueque es en Chapultepec?", "¿qué productos hay en el trueque?", "¿cómo canjeo mis puntos verdes?",
        "¿a qué hora es el mercado de trueque?", "¿hay trueque en noviembre?", "¿el trueque recibe ropa?",
        "¿necesito llevar la basura separada al trueque?", "¿puedo cambiar latas por verduras?",
        "¿dónde es el trueque de este mes?", "trueque de residuos sedema", "¿qué es eso de cambiar basura por comida?",
        "¿el mercado de trueque acepta electrónicos?", "¿cuál es el mínimo de kilos para el trueque?",
    ]
    frases = [("manual", f) for f in manuales]
    for f in random.sample(manuales, 10):
        frases.append(("plantilla", random.choice(PREFIJOS[3:]) + f))
    return frases


def normativa():
    manuales = [
        "¿qué dice la NADF-024?", "¿qué pasa si no separo mi basura?", "¿desde cuándo es obligatorio separar?",
        "¿qué ley regula la basura en CDMX?", "¿me pueden sancionar por no separar?", "¿qué norma aplica en la ciudad?",
        "¿el camión puede negarse a llevarse mi basura?", "¿por qué debo separar la basura?", "¿de verdad es obligatorio separar?",
        "¿cuál es la norma de residuos de la ciudad de méxico?", "¿hay sanciones por mezclar la basura?",
        "¿quién obliga a separar la basura?", "¿la separación de residuos es ley?", "¿cuánto es la multa por no separar?",
        "¿qué gano con separar la basura?", "¿sirve de algo separar?", "¿para qué sirve separar la basura?",
        "¿es cierto que ya no se llevan la basura revuelta?", "¿qué cambió en 2026 con la basura?",
        "¿la norma ambiental dice cuántos botes debo tener?", "¿quién publicó la norma de separación?",
        "¿qué es la NADF?", "¿el gobierno puede multarme?", "¿desde enero es obligatorio?",
        "¿los edificios también deben separar?", "¿por qué la ciudad pide separar en cuatro?",
        "¿qué beneficios tiene separar residuos?", "¿qué pasa con la basura que no se separa?",
    ]
    frases = [("manual", f) for f in manuales]
    for f in random.sample(manuales, 10):
        frases.append(("plantilla", random.choice(PREFIJOS[3:]) + f))
    return frases


def saludo():
    manuales = [
        "hola", "buenos días", "buenas tardes", "buenas noches", "qué onda", "hey", "holi", "saludos",
        "qué tal", "hola ecobot", "buenas", "hola, ¿cómo estás?", "holaaa", "qué hubo", "hola bot",
        "muy buenos días", "hola, ¿qué tal?", "buenas, ¿cómo estás?", "hola qué onda", "qué pex", "hi",
        "buenas tardes ecobot", "hola, buen día", "qué tal, buenas noches", "hola, ¿estás ahí?", "oye, hola",
        "saludos ecobot", "hey, ¿qué tal?", "hola hola", "buen día, ¿cómo te va?",
    ]
    frases = [("manual", f) for f in manuales]
    return frases


def despedida():
    manuales = [
        "adiós", "bye", "hasta luego", "nos vemos", "chao", "me voy", "hasta pronto", "eso es todo",
        "ya terminé", "salir", "cuídate", "nos vemos al rato", "sería todo", "ya me voy", "bai",
        "hasta mañana", "adiós ecobot", "bye bye", "nos vemos luego", "listo, me despido", "terminar",
        "me despido", "hasta la próxima", "chao, cuídate", "ya no tengo más dudas, adiós", "fin",
        "hasta luego bot", "bueno, me voy", "ahí nos vemos", "cerrar conversación",
    ]
    return [("manual", f) for f in manuales]


def agradecimiento():
    manuales = [
        "gracias", "muchas gracias", "te agradezco", "mil gracias", "gracias por la ayuda", "perfecto, gracias",
        "excelente", "súper", "va, gracias", "ok gracias", "thx", "muy amable", "me sirvió mucho", "genial",
        "chido, gracias", "qué buena info", "muchísimas gracias", "gracias, me ayudaste", "perfecto",
        "gracias ecobot", "te pasaste, gracias", "ok, entendido, gracias", "muy útil", "gracias por la info",
        "excelente respuesta", "qué amable, gracias", "listo, gracias", "graciaaas", "gracias bot", "sale, gracias",
    ]
    return [("manual", f) for f in manuales]


def ayuda():
    manuales = [
        "ayuda", "¿qué puedes hacer?", "¿cómo funcionas?", "¿qué sabes hacer?", "menú", "opciones",
        "¿en qué me ayudas?", "no sé qué preguntarte", "¿para qué sirves?", "¿qué te puedo preguntar?",
        "instrucciones", "¿cómo te uso?", "¿qué temas manejas?", "help", "necesito ayuda", "¿qué haces?",
        "¿cómo te pregunto algo?", "muéstrame las opciones", "¿qué tipo de preguntas respondes?",
        "¿me explicas cómo usarte?", "¿qué información tienes?", "¿con qué me puedes ayudar?",
        "dame el menú", "¿qué preguntas te puedo hacer?", "no entiendo cómo funciona esto",
        "¿puedes ver fotos?", "¿cómo te mando una imagen?", "¿qué comandos tienes?",
    ]
    return [("manual", f) for f in manuales]


def fuera_de_alcance():
    manuales = [
        "¿cómo va a estar el clima?", "¿va a llover hoy?", "¿quién ganó el clásico?", "cuéntame un chiste",
        "¿qué hora es?", "recomiéndame un restaurante", "¿cómo hago tamales?", "¿cuánto está el dólar?",
        "¿quién es el presidente?", "¿me ayudas con mi tarea de matemáticas?", "¿cómo llego al zócalo?",
        "traduce esto al inglés", "¿qué película me recomiendas?", "¿cuál es la capital de Francia?",
        "¿cómo programo en python?", "¿me duele la cabeza, qué tomo?", "¿qué es el amor?", "cántame una canción",
        "¿cuánto cuesta un iPhone?", "¿a qué hora abre el metro?", "¿cómo saco mi pasaporte?",
        "¿quién eres en realidad?", "¿tienes novia?", "escribe un poema", "¿cómo invierto mi dinero?",
        "¿qué horóscopo soy?", "¿cuántos años tienes?", "¿cómo arreglo mi computadora?",
        "¿dónde compro botes de basura baratos?", "¿cómo hago composta en un departamento grande?",
        "¿qué onda con el tráfico en periférico?", "pon música", "¿cuándo es la quincena?",
        "¿cuánto mide el Everest?", "¿me pides un uber?", "¿cómo bajo de peso?", "¿qué comida es más sana?",
        "¿cuánto cuesta el predial?", "¿cómo tramito mi licencia de conducir?", "dime una frase motivacional",
        "¿quién descubrió américa?", "¿cómo cuido un perro?", "¿qué es la inteligencia artificial?",
        "¿cuál es el mejor equipo de futbol?", "¿hoy hay hoy no circula?", "¿cómo se hace un pastel?",
        "abc xyz", "asdfgh", "jajaja", "?", "mmm", "no sé",
    ]
    frases = [("manual", f) for f in manuales]
    return frases


GENERADORES = {
    "clasificar_residuo": clasificar_residuo, "dias_recoleccion": dias_recoleccion,
    "info_categoria": info_categoria, "colores": colores, "preparacion": preparacion,
    "reciclatron": reciclatron, "mercado_trueque": mercado_trueque, "normativa": normativa,
    "saludo": saludo, "despedida": despedida, "agradecimiento": agradecimiento, "ayuda": ayuda,
    "fuera_de_alcance": fuera_de_alcance,
}


def con_error(texto: str) -> str:
    """Simula un error de tecleo: borra, duplica o intercambia una letra de una palabra larga."""
    palabras = texto.split()
    largas = [i for i, w in enumerate(palabras) if len(re.sub(r"\W", "", w)) >= 5]
    if not largas:
        return texto
    i = random.choice(largas)
    w = palabras[i]
    j = random.randrange(1, len(w) - 2)
    op = random.choice(["borrar", "duplicar", "intercambiar"])
    if op == "borrar":
        w = w[:j] + w[j + 1:]
    elif op == "duplicar":
        w = w[:j] + w[j] + w[j:]
    else:
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    palabras[i] = w
    return " ".join(palabras)


def clave(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]", "", t).strip()


# --------------------------------------------------------------------------- #
# Conjuntos de prueba
# --------------------------------------------------------------------------- #
def prueba_externa():
    """Las 57 consultas de las pruebas del Avance 1 (37 turnos + 20 de validación)."""
    sys.path.insert(0, str(RAIZ))
    from pruebas import CONVERSACIONES
    from validacion import CASOS
    filas = []
    for c in CONVERSACIONES:
        for texto, intencion, _ in c["turnos"]:
            filas.append((texto, "fuera_de_alcance" if intencion == "fallback" else intencion, "avance1_conversaciones"))
    for texto, intencion, _ in CASOS:
        filas.append((texto, "fuera_de_alcance" if intencion == "fallback" else intencion, "avance1_validacion"))
    return filas


PRUEBA_NUEVA = [
    ("¿dónde echo las bolsitas de té usadas?", "clasificar_residuo"), ("¿el bote de crema va en reciclables?", "clasificar_residuo"),
    ("traigo un montón de vasos de unicel, ¿dónde van?", "clasificar_residuo"), ("¿esto que te mandé en la foto dónde va?", "clasificar_residuo"),
    ("¿los jueves pasan por la orgánica?", "dias_recoleccion"), ("¿cuándo se llevan el cartón?", "dias_recoleccion"),
    ("¿qué día es el de la basura inorgánica?", "dias_recoleccion"), ("¿pasan el domingo?", "dias_recoleccion"),
    ("¿qué tipo de cosas son orgánicas?", "info_categoria"), ("¿qué entra en manejo especial?", "info_categoria"),
    ("¿qué diferencia hay entre los inorgánicos reciclables y los no reciclables?", "info_categoria"), ("¿qué son los voluminosos?", "info_categoria"),
    ("¿el bote verde es para orgánicos?", "colores"), ("¿qué color corresponde al bote de lo no reciclable?", "colores"),
    ("¿los colores de los contenedores qué significan?", "colores"), ("¿qué se pone en el café?", "colores"),
    ("¿hay que quitarle la tapa al garrafón?", "preparacion"), ("¿las latas se aplastan antes de entregarlas?", "preparacion"),
    ("¿tengo que limpiar los frascos?", "preparacion"), ("¿cómo dejo listo el tetrapak para reciclar?", "preparacion"),
    ("¿cuándo hay reciclatrón en la UAM?", "reciclatron"), ("¿dónde dejo una impresora vieja para reciclar?", "reciclatron"),
    ("¿el reciclatrón recibe celulares?", "reciclatron"), ("¿qué día es el reciclatrón de noviembre?", "reciclatron"),
    ("¿dónde es el próximo mercado de trueque?", "mercado_trueque"), ("¿me dan verduras si llevo reciclables?", "mercado_trueque"),
    ("¿cuántos kilos tengo que llevar al trueque?", "mercado_trueque"), ("¿qué cosas aceptan en el trueque?", "mercado_trueque"),
    ("¿qué pasa si mezclo la basura?", "normativa"), ("¿existe alguna norma para separar residuos?", "normativa"),
    ("¿es ley separar la basura en la ciudad?", "normativa"), ("¿me multan si no separo?", "normativa"),
    ("buenas tardes, ¿qué tal?", "saludo"), ("qué onda bot", "saludo"), ("hola, buenas noches", "saludo"), ("holi, ¿qué tal?", "saludo"),
    ("bueno, adiós", "despedida"), ("nos vemos mañana", "despedida"), ("ya me voy, bye", "despedida"), ("hasta luego, cuídate", "despedida"),
    ("muchas gracias por todo", "agradecimiento"), ("excelente, me sirvió", "agradecimiento"), ("va, mil gracias", "agradecimiento"), ("gracias, muy útil", "agradecimiento"),
    ("¿qué cosas sabes?", "ayuda"), ("¿cómo te pregunto?", "ayuda"), ("dame opciones", "ayuda"), ("¿me ayudas? no sé cómo usarte", "ayuda"),
    ("¿dónde venden tacos buenos?", "fuera_de_alcance"), ("¿qué hora cierra el banco?", "fuera_de_alcance"),
    ("¿cómo está el tráfico?", "fuera_de_alcance"), ("dime un chiste de programadores", "fuera_de_alcance"),
]


def main():
    filas = []
    for intencion, gen in GENERADORES.items():
        for origen, texto in gen():
            texto = texto.strip().replace(" de el ", " del ").replace(" a el ", " al ")
            filas.append((texto, intencion, origen))

    # Ruido de tecleo: 1 de cada 6 frases manuales genera una copia con error.
    manuales = [f for f in filas if f[2] == "manual"]
    for texto, intencion, _ in random.sample(manuales, len(manuales) // 6):
        filas.append((con_error(texto), intencion, "ruido"))

    # Se eliminan duplicados y cualquier frase que coincida con los conjuntos de prueba.
    externa = prueba_externa()
    claves_prueba = {clave(t) for t, _, _ in externa} | {clave(t) for t, _ in PRUEBA_NUEVA}
    vistos, limpias, quitadas = set(), [], 0
    for texto, intencion, origen in filas:
        k = clave(texto)
        if k in claves_prueba:
            quitadas += 1
            continue
        if k in vistos:
            continue
        vistos.add(k)
        limpias.append((texto, intencion, origen))
    random.shuffle(limpias)

    with open(AQUI / "intenciones_entrenamiento.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "intencion", "origen"])
        w.writerows(limpias)
    with open(AQUI / "prueba_externa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "intencion", "origen"])
        w.writerows(externa)
    with open(AQUI / "prueba_nueva.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "intencion", "origen"])
        w.writerows([(t, i, "avance2_nueva") for t, i in PRUEBA_NUEVA])

    from collections import Counter
    print(f"Entrenamiento: {len(limpias)} frases ({quitadas} descartadas por coincidir con pruebas)")
    for k, v in sorted(Counter(i for _, i, _ in limpias).items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")
    print("Origen:", dict(Counter(o for _, _, o in limpias)))
    print(f"Prueba externa (Avance 1): {len(externa)}   Prueba nueva: {len(PRUEBA_NUEVA)}")


if __name__ == "__main__":
    main()
