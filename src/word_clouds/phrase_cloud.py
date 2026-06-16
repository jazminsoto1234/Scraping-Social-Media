"""
Generador de nubes de frases/palabras por sentimiento.

Lógica desacoplada del dashboard — no depende de Streamlit.
Entrada : lista de comentarios clasificados (dicts con 'comentario' y 'clasificacion')
Salida  : figura matplotlib  o  frecuencias de frases como dict JSON-serializable
          o JSONs por video en outputs_front/<operador>/VIDEO_PHRASES_<id>.json
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Whitelist de términos relevantes (corregida: coma faltante añadida)
# ---------------------------------------------------------------------------
TELECOM_KEYWORDS: set[str] = {
    # ---- Tecnología ----
    "4g", "5g", "3g", "lte", "edge", "red", "wifi", "sim", "pin",
    "bug", "app", "sms", "mms", "qr", "datos", "llamada", "señal",
    "conexión", "internet", "móvil", "servicio", "plan", "cobertura",
    "volte", "starlink", "roaming", "llamadas", "minutos",
    # ---- Sentimiento positivo ----
    "excelente", "bueno", "funciona", "perfecto", "rápido", "confiable",
    "recomiendo", "satisfecho", "contento", "genial", "increíble",
    # ---- Sentimiento negativo ----
    "problema", "falla", "lento", "pésimo", "malo", "decepción",
    "queja", "reclamo", "estafa", "corte", "interrupciones", "cae",
    # ---- Preguntas / consultas ----
    "precio", "costo", "planes", "disponible", "cuánto", "cómo",
    "dónde", "requisitos",
    # ---- Competidores ----
    "movistar", "claro", "bitel", "entel", "viettel",
}

LABEL_COLORS: dict[str, str] = {
    "positivo":  "#22C55E",
    "negativo":  "#DC2626",
    "neutral":   "#6B7280",
    "informativo": "#7C3AED",
    "irrelevante": "#9CA3AF",
}

# ---------------------------------------------------------------------------
# Extracción de frases (bigramas + unigramas del dominio)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[\w\dáéíóúñüÁÉÍÓÚÑÜ]+\b', text.lower(), re.UNICODE)


_STOPWORDS_ES: set[str] = {
    "de", "la", "el", "en", "y", "a", "que", "los", "las", "un", "una",
    "es", "con", "por", "para", "al", "del", "se", "lo", "le", "su",
    "me", "mi", "te", "tu", "yo", "no", "si", "ya", "pero", "o", "e",
    "más", "muy", "bien", "mal", "hay", "hay", "así", "aquí", "ahí",
    "esto", "eso", "esa", "ese", "esta", "este", "era", "fue", "son",
    "han", "has", "he", "ser", "estar", "tiene", "tengo", "van", "voy",
    "como", "cuando", "donde", "porque", "aunque", "sin", "sobre",
    "todo", "todos", "cada", "solo", "sólo", "tan", "también", "ni",
}

# tamaños de n-grama a extraer (frases de 2 a 5 palabras)
_NGRAM_SIZES = (2, 3, 4, 5)


def _is_relevant_ngram(tokens: list[str]) -> bool:
    """Un n-grama es relevante si contiene al menos un término de dominio telco."""
    return any(t in TELECOM_KEYWORDS for t in tokens)


def _is_boundary(token: str) -> bool:
    """Detecta tokens que no deben estar en los extremos de una frase."""
    return token in _STOPWORDS_ES or len(token) <= 1


def _clean_ngram(tokens: list[str]) -> list[str]:
    """Recorta stopwords de los extremos del n-grama."""
    while tokens and _is_boundary(tokens[0]):
        tokens = tokens[1:]
    while tokens and _is_boundary(tokens[-1]):
        tokens = tokens[:-1]
    return tokens


def _split_into_clauses(text: str) -> list[list[str]]:
    """
    Divide el texto en cláusulas por puntuación y conjunciones adversativas,
    tokeniza cada cláusula. Evita frases que crucen límites de oración.
    """
    clauses = re.split(r'[.,;:!?\n]|(?:\bpero\b|\baunque\b|\bsin embargo\b)', text.lower())
    return [_tokenize(c) for c in clauses if c.strip()]


def _extract_phrases(texts: list[str], max_phrases: int = 40) -> dict[str, int]:
    """
    Extrae frases de longitud variable (2-5 palabras) de cada comentario.
    - Se divide por cláusulas para no cruzar límites de oración
    - Solo pasan n-gramas con al menos un término de dominio telco
    - Se recortan stopwords en los extremos
    - Frases largas (3-5 tokens) tienen prioridad sobre bigramas
    Devuelve {frase: frecuencia} ordenado por frecuencia desc.
    """
    counts_by_size: dict[int, Counter] = {n: Counter() for n in _NGRAM_SIZES}

    for text in texts:
        if not text or not text.strip():
            continue
        for clause_tokens in _split_into_clauses(text):
            if len(clause_tokens) < 2:
                continue
            for n in _NGRAM_SIZES:
                for i in range(len(clause_tokens) - n + 1):
                    ngram = clause_tokens[i: i + n]
                    if not _is_relevant_ngram(ngram):
                        continue
                    cleaned = _clean_ngram(ngram[:])
                    if len(cleaned) < 2:
                        continue
                    counts_by_size[n][" ".join(cleaned)] += 1

    # prioridad: frases más largas primero (más contexto), luego bigramas
    merged: dict[str, int] = {}
    for n in reversed(_NGRAM_SIZES):
        for phrase, count in counts_by_size[n].most_common():
            if count < 2:
                continue
            # evitar subfrases ya cubiertas por una frase más larga
            if any(phrase in longer for longer in merged):
                continue
            merged[phrase] = count
            if len(merged) >= max_phrases:
                break
        if len(merged) >= max_phrases:
            break

    # completar con unigramas telco si hay espacio
    if len(merged) < max_phrases:
        unigram_counts: Counter = Counter()
        for text in texts:
            for tok in _tokenize(text):
                if tok in TELECOM_KEYWORDS and tok not in _STOPWORDS_ES:
                    unigram_counts[tok] += 1
        for tok, count in unigram_counts.most_common():
            if tok not in " ".join(merged.keys()):
                merged[tok] = count
            if len(merged) >= max_phrases:
                break

    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True)[:max_phrases])


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_phrase_frequencies(
    comments: list[dict],
    sentiment: Literal["positivo", "negativo", "neutral", "informativo", "irrelevante"],
    text_field: str = "comentario",
    max_phrases: int = 40,
) -> dict[str, int]:
    """
    Devuelve {frase: frecuencia} para los comentarios del sentimiento indicado.
    JSON-serializable directamente.
    """
    texts = [
        c.get(text_field, "") or ""
        for c in comments
        if c.get("clasificacion", "").lower() == sentiment
    ]
    return _extract_phrases(texts, max_phrases=max_phrases)


def build_phrase_cloud_figure(
    comments: list[dict],
    sentiment: Literal["positivo", "negativo", "neutral", "informativo", "irrelevante"],
    text_field: str = "comentario",
    max_phrases: int = 40,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure | None:
    """
    Genera y devuelve una figura matplotlib con la nube de frases.
    Devuelve None si no hay suficiente texto.
    """
    freqs = get_phrase_frequencies(comments, sentiment, text_field, max_phrases)
    if not freqs:
        return None

    color = LABEL_COLORS.get(sentiment, "#6B7280")

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        color_func=lambda *_, **__: color,
        stopwords=set(),
        max_words=max_phrases,
        collocations=False,
        prefer_horizontal=0.85,
        min_word_length=2,
    ).generate_from_frequencies(freqs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
# Generación de JSONs para frontend
# ---------------------------------------------------------------------------

_SENTIMENTS = ("positivo", "negativo", "neutral", "informativo", "irrelevante")
_ROOT = Path(__file__).resolve().parents[2]


def generate_phrase_jsons_for_operator(
    operador: str,
    classified_dir: Path = None,
    front_dir: Path = None,
    text_field: str = "comentario",
    max_phrases: int = 40,
) -> list[Path]:
    """
    Lee VIDEO_CLASSIFIED_*.json y facebook_classified_*.json de
    outputs/<operador>/clasificado/ y genera:
      - outputs_front/<operador>/VIDEO_PHRASES_<id>.json   (por TikTok video)
      - outputs_front/<operador>/FB_PHRASES_<id>.json      (por Facebook post)
      - outputs_front/<operador>/GENERAL_PHRASES.json      (todos los posts del mes)

    Devuelve lista de paths escritos.
    """
    if classified_dir is None:
        classified_dir = _ROOT / "outputs" / operador / "clasificado"
    if front_dir is None:
        front_dir = _ROOT / "outputs_front" / operador

    front_dir.mkdir(parents=True, exist_ok=True)

    # (glob_pattern, id_field, in_prefix, out_prefix)
    _PATTERNS = [
        ("VIDEO_CLASSIFIED_*.json",   "video_id", "VIDEO_CLASSIFIED_", "VIDEO_PHRASES_"),
        ("facebook_classified_*.json", "post_id",  "facebook_classified_", "FB_PHRASES_"),
    ]

    escritos = []
    all_comments: list[dict] = []

    for glob_pat, id_field, in_prefix, out_prefix in _PATTERNS:
        for archivo in sorted(classified_dir.glob(glob_pat)):
            try:
                data = json.loads(archivo.read_text(encoding="utf-8"))
            except Exception:
                continue

            item_id = data.get(id_field, archivo.stem.replace(in_prefix, ""))
            comments = data.get("comentarios", [])
            all_comments.extend(comments)

            resultado: dict = {
                id_field:     item_id,
                "operador":   data.get("operador", operador),
                "plataforma": data.get("plataforma", "tiktok"),
                "fecha":      data.get("fecha_video") or data.get("fecha_post", ""),
            }
            for sentiment in _SENTIMENTS:
                freqs = get_phrase_frequencies(comments, sentiment, text_field, max_phrases)
                if freqs:
                    resultado[sentiment] = freqs

            out_path = front_dir / f"{out_prefix}{item_id}.json"
            out_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
            escritos.append(out_path)

    # JSON general: agrega todos los comentarios del mes (TikTok + Facebook)
    if all_comments:
        general: dict = {"operador": operador, "total_comentarios": len(all_comments)}
        for sentiment in _SENTIMENTS:
            freqs = get_phrase_frequencies(all_comments, sentiment, text_field, max_phrases)
            if freqs:
                general[sentiment] = freqs
        out_general = front_dir / "GENERAL_PHRASES.json"
        out_general.write_text(json.dumps(general, ensure_ascii=False, indent=2), encoding="utf-8")
        escritos.append(out_general)

    return escritos


def generate_phrase_jsons_all_operators(
    operators: list[str] = None,
    base_dir: Path = None,
    front_dir: Path = None,
    **kwargs,
) -> dict[str, list[Path]]:
    """
    Corre generate_phrase_jsons_for_operator para todos los operadores.
    Si operators es None, detecta automáticamente las carpetas con /clasificado/.
    Devuelve {operador: [paths escritos]}.
    """
    if base_dir is None:
        base_dir = _ROOT / "outputs"

    if operators is None:
        operators = [
            d.name for d in base_dir.iterdir()
            if d.is_dir() and (d / "clasificado").exists()
        ]

    return {
        op: generate_phrase_jsons_for_operator(op, front_dir=front_dir, **kwargs)
        for op in sorted(operators)
    }
