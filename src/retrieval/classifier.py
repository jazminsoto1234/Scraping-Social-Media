"""
Pipeline de sentimiento en 3 capas:
  Capa 1 — RoBERTuito (pysentimiento): positivo/negativo/neutro
  Capa 2 — Detector de sarcasmo independiente (RoBERTuito fine-tuned o GPT-4o-mini few-shot)
  Capa 3 — Fusión: sarcasmo+positivo → negativo, sarcasmo+neutro → negativo
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pysentimiento import create_analyzer

from transformers import pipeline as hf_pipeline


logger = logging.getLogger(__name__)

_LABELS = ["positivo", "negativo", "informativo", "neutral", "irrelevante"]

# ---------------------------------------------------------------------------
# Capa 1: RoBERTuito via pysentimiento
# ---------------------------------------------------------------------------

_sentiment_analyzer = None


def _get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
            logger.info("RoBERTuito cargado correctamente")
        except Exception as e:
            logger.warning("pysentimiento no disponible (%s) — fallback a TF-IDF", e)
            _sentiment_analyzer = "fallback"
    return _sentiment_analyzer


def _classify_sentiment_roberta(text: str) -> tuple[str, float]:
    """Devuelve (label, confianza) usando RoBERTuito."""
    analyzer = _get_sentiment_analyzer()
    if analyzer == "fallback":
        return _classify_sentiment_tfidf(text), 0.5

    try:
        result = analyzer.predict(text)
        label_map = {"POS": "positivo", "NEG": "negativo", "NEU": "neutral"}
        label = label_map.get(result.output, "neutral")
        confidence = result.probas.get(result.output, 0.5)
        return label, confidence
    except Exception as e:
        logger.debug("Error en RoBERTuito: %s", e)
        return "neutral", 0.5


# ---------------------------------------------------------------------------
# Fallback TF-IDF (igual que antes, para cuando pysentimiento no está)
# ---------------------------------------------------------------------------

import nltk
from nltk.stem import RSLPStemmer
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("rslp", quiet=True)
_stemmer = RSLPStemmer()


def _normalize(text: str) -> str:
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return " ".join(_stemmer.stem(w) for w in text.split())


_SEED_DATA: dict[str, list[str]] = {
    "positivo": [
        "me encantó", "excelente servicio", "muy bueno", "recomendado", "genial",
        "perfecto", "increíble", "feliz", "satisfecho", "gracias",
        "me gustó mucho", "lo mejor", "super bueno", "buenísimo",
        "maravilloso", "espectacular", "funciona perfecto", "muy contento",
        "gran servicio", "muy satisfecho", "amable", "atento",
        "buena señal", "buena cobertura", "funciona bien", "fluido", "rápido",
        "impecable", "confiable", "internet rápido", "excelente atención", "muy amable",
        "me resolvieron", "estable ahora", "mejor internet", "recomendadísimo",
        "buen precio", "técnico rápido", "buena promoción", "me ayudaron",
        "servicio excelente", "muy satisfecho", "económico", "buena velocidad",
        "sin problemas", "mejora notable", "eficiente", "salvador",
        "definitivamente lo mejor", "me salvó", "excelente cobertura",
        "funciona perfectamente", "lo recomiendo", "estoy contento", "estoy feliz",
        "llega bien rápido", "conexión perfecta", "se salvó cuando necesitaba",
        "no me arrepiento de cambiar de claro a entel", "no me arrepiento del cambio",
        "con señal satelital no me preocupo", "señal satelital es perfecta",
        "me funciona perfectamente", "muy buenísimo", "buenísimo el servicio",
        "solo tienen que mejorar", "espero que mejoren", "falta mejorar pero es bueno",
    ],
    "negativo": [
        "pésimo", "pésima", "no sirve", "muy malo", "decepcionante", "horrible",
        "una estafa", "mala atención", "no funciona correctamente", "terrible", "fatal",
        "queja", "reclamo", "me cobraron de más", "problema grave",
        "no recomiendo", "peor servicio", "desastre", "inaceptable",
        "me han fallado", "muy decepcionado", "falla", "lento", "problemas",
        "caída", "caído", "fallando", "sin señal", "interrupción", "robo", "caro", "saturado",
        "inestable", "mal servicio",
        "no responden", "nunca contestan", "cortes", "lag", "ping alto", "fraude",
        "basura", "indignado",
        "cancelaré", "deficiente", "caída masiva", "internet lento",
        "llamada caída", "cobertura limitada", "no solucionan",
        "demora", "espera eterna", "cola", "desconectado",
        "caídas frecuentes", "buffering", "congelado", "trabado",
        "sin internet", "sin cobertura", "servicio horrible",
        "nunca llega el técnico", "señal débil", "lento en la noche",
        "microcortes", "soporte inútil", "mala facturación",
        "cobro indebido", "caída nacional", "saturación", "desconexión",
        "peor operador", "frustrante", "incompetentes", "demasiado caro",
        "técnico nunca llegó", "entel nunca más", "cancelaré mi línea",
        "daré de baja", "portabilidad ya",
        "por eso me cambio", "por eso me paso", "por eso prefiero",
        "me cambio a claro", "me paso a claro", "voy a claro", "mejor claro",
        "me cambio a movistar", "me paso a movistar", "voy a movistar", "mejor movistar",
        "me cambio a bitel", "me paso a bitel", "voy a bitel", "mejor bitel",
        "me voy a claro", "me voy a movistar", "me voy a bitel",
        "prefiero claro", "prefiero movistar", "prefiero bitel",
        "me regreso a claro", "me regreso a movistar", "me regreso a bitel",
        "mala señal", "mejor con claro", "mejor con movistar", "mejor con bitel",
        "no compatible", "no funciona con", "no me deja", "no carga", "no deja",
        "problema con", "sigue fallando", "falla permanente", "no mejoró",
        "gastazo", "me estafaron", "no vale la pena", "no funciona nada",
        "pésima cobertura", "conexión muere", "desconexión constante",
        "error constante", "servicio pésimo", "nunca funciona", "totalmente inútil",
        "es malo", "es pésimo", "mal servicio", "muy malo", "malo",
        "no funciona", "es lento", "no tiene cobertura",
        "no me dan solución", "no funciona la conexión",
        "no llega la señal", "no va bien",
        "no recomendable", "evitar", "no pasen",
    ],
    "informativo": [
        "quiero saber", "me pueden explicar", "cuánto cuesta",
        "cómo funciona", "cómo activo", "cómo contrato",
        "cuáles son los planes", "qué incluye", "tienen disponible",
        "dónde comprar", "dónde contrato", "requisitos para",
        "cuándo disponible", "a qué hora", "cuántos días demora",
        "precio", "costos", "promoción vigente",
        "información sobre", "me interesa", "info", "más info",
        "necesito información", "pasa info", "link por favor",
        "cobertura en mi zona", "técnico", "instalación",
        "cuál es el costo", "promoción actual", "planes hogar",
        "planes móviles", "quiero migrar", "quiero portar",
        "asesor", "dm", "escriban al inbox", "interno",
        "cómo saber", "como saber", "cómo configuro", "como configuro",
        "quiero adquirir", "deseo contratar", "me interesa contratar",
        "es con starlink", "es para zonas", "para qué zonas",
        "es compatible con", "funciona en", "llega en",
    ],
    "neutral": [
        "ok", "vale", "oke", "normal", "está bien",
        "lo vi", "interesante", "tal vez", "puede ser",
        "depende", "ya vi", "entendido", "de acuerdo",
        "vamos a ver", "habrá que esperar", "no sé", "quizás",
        "es verdad", "es cierto", "jaja", "jajaja",
        "trending", "trend", "meme", "jajajaja", "xd",
        "hola", "buenas tardes", "buenos días",
    ],
}

_DEFAULT_MODEL_PATH = Path("outputs/modelo_clasico.pkl")
_tfidf_pipeline: Optional[Pipeline] = None


def _get_tfidf_pipeline() -> Pipeline:
    global _tfidf_pipeline
    if _tfidf_pipeline is not None:
        return _tfidf_pipeline
    if _DEFAULT_MODEL_PATH.exists():
        _tfidf_pipeline = joblib.load(_DEFAULT_MODEL_PATH)
    else:
        texts, labels = [], []
        for label, phrases in _SEED_DATA.items():
            for phrase in phrases:
                texts.append(_normalize(phrase))
                labels.append(label)
        pipeline = Pipeline([
            ("tfidf", _TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced")),
        ])
        pipeline.fit(texts, labels)
        _DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, _DEFAULT_MODEL_PATH)
        _tfidf_pipeline = pipeline
    return _tfidf_pipeline


def _classify_sentiment_tfidf(text: str) -> str:
    pipeline = _get_tfidf_pipeline()
    normalized = _normalize(text)
    if not normalized.strip():
        return "neutral"
    return pipeline.predict([normalized])[0]


# ---------------------------------------------------------------------------
# Reglas estructurales (preguntas, elogios, quejas inequívocas)
# ---------------------------------------------------------------------------

_DOMAIN_LEXICON: set[str] = {
    "señal", "senal", "cobertura", "internet", "wifi", "datos", "data",
    "plan", "planes", "recarga", "recargas", "megas", "mega", "gigas",
    "velocidad", "lento", "lenta", "rápido", "rapido", "conexión", "conexion",
    "red", "antena", "antenas", "satelital", "starlink", "5g", "4g", "3g",
    "llamada", "llamadas", "chip", "linea", "línea", "número", "numero",
    "factura", "facturación", "facturacion", "cobro", "cobraron", "tarifa",
    "portabilidad", "portar", "migrar", "técnico", "tecnico", "instalación",
    "instalacion", "soporte", "atención", "atencion", "servicio", "operador",
    "celular", "teléfono", "telefono", "equipo", "wasap", "whatsapp",
    "saturado", "saturación", "saturacion", "corte", "cortes", "caída",
    "caida", "buffering", "lag", "ping", "promoción", "promocion", "precio",
    "roaming", "minutos", "sms",
    "entel", "claro", "movistar", "bitel", "empresa", "compañía", "compania",
    "avería", "averia", "postpago", "prepago", "publicidad", "solución",
    "solucion", "problema", "reclamo", "queja", "irme", "cambiarme",
}

_QUESTION_STARTERS = (
    "cuánto", "cuanto", "cómo", "como", "dónde", "donde", "qué", "que",
    "cuál", "cual", "cuándo", "cuando", "hay ", "tienen", "tendrán",
    "tendran", "alguien sabe", "se puede", "puedo",
)

_STRONG_POSITIVE = (
    "lo máximo", "lo maximo", "todo normal", "lo recomiendo",
    "la recomiendo", "muy recomendable", "funciona perfecto", "funciona bien",
    "excelente señal", "excelente cobertura", "buena señal", "buena cobertura",
    "me encanta", "me encantó", "me encanto", "lo mejor", "súper bien",
    "super bien", "anda bien", "anda perfecto", "sin problemas",
    "vaz con todo", "va con todo", "amo entel", "amo claro",
    "amo movistar", "amo bitel", "entel mi familia", "mi familia entel",
    "me saluda me cambio", "me cambio con entel",
    "no me arrepiento de cambiar", "no me arrepiento del cambio",
    "señal satelital no me preocupo", "con señal satelital",
    "me funciona perfectamente", "funciona perfectamente",
    "muy buenisimo", "muy buenísimo", "buenísimo", "buenisimo",
)

_STRONG_NEGATIVE = (
    "mierda", "mrd", "pésimo", "pesimo", "pésima", "pesima", "horrible",
    "no sirve", "basura", "una estafa", "publicidad falsa", "publicidad engañosa",
    "es un chiste", "no recomiendo", "no lo recomiendo", "no migren", "asco",
    "nunca solucionan", "no me dan solución", "no me dan solucion",
    "recontra lento", "más pésimo", "mas pesimo", "no solucionan",
    "sin solución", "sin solucion", "avería", "averia",
    "más de 2 horas sin", "más de 2 horas y no",
    "no recomendable", "solo en la ciudad", "solo ciudad",
    "prefiero bitel", "prefiero claro", "prefiero movistar",
    "bitel es mejor", "claro es mejor", "movistar es mejor",
    "mejor bitel", "mejor claro", "mejor movistar",
    "es mejor que entel", "son mejores que entel",
    "peor operador", "el peor operador",
    "nunca tienen buena señal", "nunca hay buena señal",
    "nunca tienen cobertura", "no hay buena señal",
)


def _has_domain_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _DOMAIN_LEXICON)


def _apply_rules(text: str, predicted: str) -> str:
    lowered = text.lower().strip()
    if not lowered:
        return predicted

    is_question = "?" in lowered or "¿" in lowered or any(
        lowered.startswith(s) for s in _QUESTION_STARTERS
    ) or lowered.startswith("una pregunta")
    has_insult = any(w in lowered for w in _STRONG_NEGATIVE)
    if is_question and not has_insult:
        return "informativo"

    _NEGATIONS = ("no ", "nunca ", "tampoco ", "ni ", "sin ", "jamás ", "jamas ")
    _CLAUSE_SPLIT = re.compile(r"[.,;!?\n]|pero |aunque |sin embargo |a pesar ")

    def _clause_of(phrase: str) -> str:
        idx = lowered.find(phrase)
        if idx == -1:
            return ""
        fragments = _CLAUSE_SPLIT.split(lowered)
        pos = 0
        for frag in fragments:
            if pos <= idx < pos + len(frag) + 1:
                return frag
            pos += len(frag) + 1
        return lowered

    def _is_negated(phrase: str) -> bool:
        clause = _clause_of(phrase)
        if not clause:
            return False
        phrase_start = clause.find(phrase)
        if phrase_start == -1:
            return False
        window = clause[max(0, phrase_start - 40):phrase_start]
        for neg in _NEGATIONS:
            if neg not in window:
                continue
            if neg == "sin " and phrase.startswith("sin"):
                continue
            return True
        return False

    if predicted == "positivo" and any(
        phrase in lowered and _is_negated(phrase) for phrase in _STRONG_POSITIVE
    ):
        return "negativo"

    # _STRONG_POSITIVE solo refuerza si RoBERTuito ya dijo positivo,
    # no sobreescribe neutral/negativo (evita falsos positivos con frases fuera de contexto)
    if predicted == "positivo" and any(
        phrase in lowered and not _is_negated(phrase) for phrase in _STRONG_POSITIVE
    ):
        return "positivo"

    if has_insult:
        return "negativo"

    return predicted


# ---------------------------------------------------------------------------
# Capa 2: Detector de sarcasmo
# ---------------------------------------------------------------------------

_sarcasm_analyzer = None

# Few-shot examples para GPT-4o-mini (solo se usan en fallback de API)
_SARCASM_EXAMPLES = [
    ("Sí claro, excelente señal, solo se cae 10 veces al día 🙄", True),
    ("Qué maravilloso servicio, llevo 3 días sin internet 👏", True),
    ("Genial la atención, tardaron solo 2 semanas en responder", True),
    ("Muy buena señal, gracias por el servicio", False),
    ("Terrible, nunca funciona el internet", False),
    ("Excelente cobertura en toda la ciudad", False),
]


def _get_sarcasm_analyzer():
    """Carga el modelo de sarcasmo fine-tuned si está disponible."""
    global _sarcasm_analyzer
    if _sarcasm_analyzer is not None:
        return _sarcasm_analyzer

    # Intentar cargar modelo fine-tuned local primero
    sarcasm_model_path = Path("outputs/sarcasm_model")
    if sarcasm_model_path.exists():
        try:
            _sarcasm_analyzer = hf_pipeline(
                "text-classification",
                model=str(sarcasm_model_path),
                device=-1,  # CPU por defecto
            )
            logger.info("Modelo de sarcasmo fine-tuned cargado desde %s", sarcasm_model_path)
            return _sarcasm_analyzer
        except Exception as e:
            logger.warning("Error cargando modelo de sarcasmo local: %s", e)

    # Intentar pysentimiento irony si está disponible
    try:
        _sarcasm_analyzer = create_analyzer(task="irony", lang="es")
        logger.info("Detector de ironía/sarcasmo de pysentimiento cargado")
        return _sarcasm_analyzer
    except Exception as e:
        logger.warning("pysentimiento irony no disponible: %s", e)

    _sarcasm_analyzer = "api_fallback"
    return _sarcasm_analyzer


def _detect_sarcasm_heuristic(text: str) -> tuple[bool, float]:
    """
    Heurística rápida como pre-filtro: detecta patrones de sarcasmo obvios
    (elogios seguidos de queja, emojis de burla, adverbios irónicos).
    Devuelve (es_sarcasmo, confianza).
    """
    lowered = text.lower()

    sarcasm_patterns = [
        # elogio + pero/aunque + queja
        r'(excelente|genial|maravillos|perfecto|increíble|buenísimo).{0,60}(pero|aunque|sin embargo|solo que|excepto).{0,60}(falla|lento|cae|corte|sin señal|sin internet)',
        # "solo" + número alto + problema frecuente
        r'solo (se cae|falla|corta|pierde señal|queda sin) \d+',
        # "qué + elogio, llevo X sin servicio"
        r'qué (bien|bueno|genial|maravillo|excelente).{0,80}(llevo|llevamos|hace \d+).{0,40}sin (internet|señal|servicio)',
        # elogios con 👏 o 🙄 (frecuentes en sarcasmo)
        r'(excelente|genial|perfecto|maravillo|increíble).{0,100}[👏🙄😒😑]',
        # "claro que sí" + negación implícita
        r'claro que sí.{0,60}(no|nunca|jamás|tampoco)',
        # "tardaron solo" + tiempo largo
        r'tardaron solo (1|2|3|4|5|6|7|8|9|10|\w+) (semana|mes|día|hora)',
    ]

    for pattern in sarcasm_patterns:
        if re.search(pattern, lowered):
            return True, 0.75

    # Emojis de burla cerca de elogio
    burla_emojis = ["🙄", "😒", "😑", "🤦", "🤣", "💀"]
    elogios = ["excelente", "genial", "perfecto", "maravilloso", "increíble", "buenísimo"]
    has_elogio = any(e in lowered for e in elogios)
    has_burla = any(em in text for em in burla_emojis)
    if has_elogio and has_burla:
        return True, 0.7

    return False, 0.0


def _detect_sarcasm(text: str, sentiment_label: str, sentiment_confidence: float) -> tuple[bool, float]:
    """
    Detecta sarcasmo en el texto. Solo se activa agresivamente cuando
    sentiment_label == 'positivo' (donde el sarcasmo más confunde).

    Devuelve (es_sarcasmo, confianza).
    """
    # Pre-filtro heurístico rápido (sin modelo)
    heuristic_result, heuristic_conf = _detect_sarcasm_heuristic(text)
    if heuristic_result and heuristic_conf >= 0.7:
        return True, heuristic_conf

    # Solo invocar modelo pesado para positivos con alta confianza
    # (es donde el sarcasmo más daña la precisión)
    if sentiment_label != "positivo" or sentiment_confidence < 0.6:
        return heuristic_result, heuristic_conf

    analyzer = _get_sarcasm_analyzer()

    if analyzer == "api_fallback":
        return _detect_sarcasm_api(text)

    try:
        # pysentimiento irony analyzer
        if hasattr(analyzer, 'predict'):
            result = analyzer.predict(text)
            is_irony = result.output == "ironic"
            confidence = result.probas.get(result.output, 0.5)
            return is_irony, confidence

        # transformers pipeline (modelo fine-tuned local)
        result = analyzer(text, truncation=True, max_length=128)[0]
        label = result["label"].upper()
        score = result["score"]
        is_sarcasm = label in ("SARCASM", "IRONIC", "LABEL_1")
        return is_sarcasm, score

    except Exception as e:
        logger.debug("Error en detector de sarcasmo: %s", e)
        return heuristic_result, heuristic_conf


def _detect_sarcasm_api(text: str) -> tuple[bool, float]:
    """Gemini few-shot como último recurso (solo si GEMINI_API_KEY está configurada)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return False, 0.0

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        examples_text = "\n".join(
            f'Texto: "{t}"\nSarcasmo: {"Sí" if s else "No"}'
            for t, s in _SARCASM_EXAMPLES
        )
        prompt = (
            "Eres un detector de sarcasmo en comentarios de redes sociales en español latinoamericano. "
            "Responde solo 'Sí' o 'No'.\n\n"
            f"Ejemplos:\n{examples_text}\n\n"
            f'Texto: "{text[:300]}"\nSarcasmo:'
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = response.text.strip().lower()
        return answer.startswith("sí") or answer.startswith("si"), 0.85

    except Exception as e:
        logger.debug("Error en Gemini sarcasmo: %s", e)
        return False, 0.0


# ---------------------------------------------------------------------------
# Capa 3: Fusión
# ---------------------------------------------------------------------------

def _fuse(sentiment: str, is_sarcasm: bool) -> str:
    """
    Lógica de fusión:
      sarcasmo + positivo    → negativo
      sarcasmo + neutral     → negativo
      sarcasmo + informativo → negativo
      sarcasmo + negativo    → negativo (sin cambio)
      sin sarcasmo           → sentimiento base sin cambio
    """
    if not is_sarcasm:
        return sentiment
    if sentiment in ("positivo", "neutral", "informativo"):
        return "negativo"
    return sentiment


# ---------------------------------------------------------------------------
# Clasificador principal
# ---------------------------------------------------------------------------

class CommentClassifier:
    """
    Clasificador de comentarios con pipeline de 3 capas:
      1. RoBERTuito (pysentimiento) para sentimiento base
      2. Detector de sarcasmo independiente
      3. Fusión con lógica de negación
    """

    def __init__(self, config: dict):
        self.cfg = config.get("retrieval", {})
        self._search_index: Optional[dict] = None

    def classify_comments(
        self,
        comments: list[dict],
        text_field: str = "comentario",
        platform: str = "tiktok",
    ) -> list[dict]:
        classified = []
        counts: dict[str, int] = {lbl: 0 for lbl in _LABELS}
        sarcasm_flips = 0
        rule_fixes = 0

        for comment in comments:
            raw = comment.get(text_field, "") or ""

            # Vacío → neutral directo
            if not raw.strip():
                out = dict(comment)
                out["clasificacion"] = "neutral"
                out["sarcasmo_detectado"] = False
                classified.append(out)
                counts["neutral"] += 1
                continue

            # Capa 1: sentimiento base
            sentiment, confidence = _classify_sentiment_roberta(raw)

            # Filtro de dominio: positivo/negativo off-topic → irrelevante
            # (excepto insulto explícito o elogio fuerte)
            if sentiment in ("positivo", "negativo"):
                has_insult = any(w in raw.lower() for w in _STRONG_NEGATIVE)
                has_strong_pos = any(p in raw.lower() for p in _STRONG_POSITIVE)
                if not _has_domain_term(raw) and not has_insult and not has_strong_pos:
                    sentiment = "irrelevante"

            # neutral sin término de dominio → irrelevante
            if sentiment == "neutral" and not _has_domain_term(raw):
                sentiment = "irrelevante"

            # Reglas estructurales (preguntas, elogios/quejas inequívocas)
            fixed = _apply_rules(raw, sentiment)
            if fixed != sentiment:
                rule_fixes += 1
            sentiment = fixed

            # Capa 2: detector de sarcasmo (heurística para todos; modelo solo para positivos)
            is_sarcasm = False
            if sentiment == "informativo":
                # solo heurística rápida, sin invocar modelos pesados
                is_sarcasm, _ = _detect_sarcasm_heuristic(raw)
            elif sentiment != "irrelevante":
                is_sarcasm, _ = _detect_sarcasm(raw, sentiment, confidence)

            # Capa 3: fusión
            final = _fuse(sentiment, is_sarcasm)
            if final != sentiment:
                sarcasm_flips += 1

            out = dict(comment)
            out["clasificacion"] = final
            out["sarcasmo_detectado"] = is_sarcasm
            classified.append(out)
            counts[final if final in counts else "neutral"] += 1

        if sarcasm_flips:
            logger.info("Fusión sarcasmo: %d comentarios reclasificados", sarcasm_flips)
        if rule_fixes:
            logger.info("Reglas post-proceso: %d comentarios reclasificados", rule_fixes)

        logger.info(
            "Clasificacion: +%d -%d info:%d neutral:%d irrelevante:%d",
            counts["positivo"], counts["negativo"], counts["informativo"],
            counts["neutral"], counts["irrelevante"],
        )

        # Construir índice TF-IDF en memoria para search_similar
        self._build_search_index(classified, text_field)
        return classified

    def _build_search_index(self, classified: list[dict], text_field: str):
        texts = [_normalize(c.get(text_field, "") or "") for c in classified]
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return
        try:
            pipeline = _get_tfidf_pipeline()
            tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
            self._search_index = {
                "matrix": tfidf.transform(texts),
                "comments": classified,
                "text_field": text_field,
            }
        except Exception as e:
            logger.warning("No se pudo construir índice de búsqueda: %s", e)

    def search_similar(self, query: str, platform: str = "tiktok", top_k: int = 10) -> list[dict]:
        if self._search_index is None:
            logger.warning(
                "search_similar: no hay índice en memoria. "
                "Ejecuta classify_comments primero."
            )
            return []

        pipeline = _get_tfidf_pipeline()
        tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
        query_vec = tfidf.transform([query])
        sims = cosine_similarity(query_vec, self._search_index["matrix"])[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        text_field = self._search_index.get("text_field", "comentario")
        results = []
        for idx in top_indices:
            comment = self._search_index["comments"][idx]
            results.append({
                "texto": comment.get(text_field, ""),
                "clasificacion": comment.get("clasificacion", ""),
                "distancia": float(1.0 - sims[idx]),
            })
        return results

    def save_embeddings(self, comments=None, output_dir=None, text_field="comentario"):
        logger.debug("save_embeddings: no-op en el clasificador de 3 capas")


def train_and_save(model_path: Path = _DEFAULT_MODEL_PATH) -> Pipeline:
    """Entrena y serializa el pipeline TF-IDF de fallback."""
    pipeline = _get_tfidf_pipeline()
    joblib.dump(pipeline, model_path)
    return pipeline
