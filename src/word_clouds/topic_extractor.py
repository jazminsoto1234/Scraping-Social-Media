"""
Extracción de topics semánticos para nubes de palabras.

Pipeline:
  1. Embeddings locales (sentence-transformers) — sin API, sin internet.
  2. UMAP + HDBSCAN — agrupa comentarios por significado, descarta ruido.
  3. Naming con Gemini 2.0 flash — nombra cada cluster en lenguaje natural.
     Fallback offline: término más frecuente del cluster si la API falla.

Salida: GENERAL_TOPICS.json en outputs_front/<operador>/
  {
    "operador": "entel",
    "total_comentarios": 3052,
    "<sentimiento>": {
      "<nombre topic>": {
        "frecuencia": 48,
        "representativas": ["frase real 1", "frase real 2"]
      }
    }
  }

La función build_topic_cloud_figure() devuelve Figure matplotlib con
  {topic_name: frecuencia}, mismo formato que phrase_cloud — el render del
  dashboard no cambia.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_SENTIMENTS = ("positivo", "negativo", "neutral", "informativo", "irrelevante")

LABEL_COLORS: dict[str, str] = {
    "positivo":    "#22C55E",
    "negativo":    "#DC2626",
    "neutral":     "#6B7280",
    "informativo": "#7C3AED",
    "irrelevante": "#9CA3AF",
}

# ─── modelo de embeddings ────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2: soporta español nativamente,
# ya disponible en sentence-transformers, ~117MB, muy rápido en CPU.
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embed_model_instance = None


def _get_embed_model():
    global _embed_model_instance
    if _embed_model_instance is None:
        from sentence_transformers import SentenceTransformer
        _embed_model_instance = SentenceTransformer(_EMBED_MODEL)
    return _embed_model_instance


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_classified_comments(operador: str) -> list[dict]:
    """Lee todos los JSONs clasificados (TikTok + Facebook) del operador."""
    classified_dir = _ROOT / "outputs" / operador / "clasificado"
    if not classified_dir.exists():
        return []
    comments: list[dict] = []
    for path in classified_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data.get("comentarios", []):
                c.setdefault("operador", operador)
                comments.append(c)
        except Exception:
            continue
    return comments


def _clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]", " ", text)
    return " ".join(text.split()).strip()


# ─── etapa 1: embeddings + clustering ────────────────────────────────────────

def _cluster_comments(texts: list[str], min_cluster: int = 3) -> list[int]:
    """
    Devuelve etiquetas de cluster por índice.
    -1 = ruido (descartado de la nube).
    Requiere al menos 2*min_cluster textos para intentar clustering.
    """
    import numpy as np

    if len(texts) < min_cluster * 2:
        # muy pocos comentarios: todos en un único cluster
        return [0] * len(texts)

    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

    # UMAP: reduce a 5 dimensiones antes de clustering
    try:
        import umap
        n_neighbors = min(15, len(texts) - 1)
        reducer = umap.UMAP(
            n_components=5,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)
    except Exception as e:
        logger.warning("UMAP falló (%s), usando embeddings directos.", e)
        reduced = embeddings

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced).tolist()
    except Exception as e:
        logger.warning("HDBSCAN falló (%s), agrupación trivial.", e)
        labels = [0] * len(texts)

    return labels


# ─── etapa 2: naming con OpenRouter (Llama 3.3 70B) ─────────────────────────

_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
_OPENROUTER_BASE  = "https://openrouter.ai/api/v1"


def _name_cluster_llm(texts_sample: list[str], sentiment: str) -> str | None:
    """
    Llama a Llama 3.3 70B vía OpenRouter para obtener un nombre de topic
    corto (3-6 palabras). Retorna None si falla o no hay key.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=_OPENROUTER_BASE)
        muestra = "\n".join(f"- {t[:120]}" for t in texts_sample[:12])
        prompt = (
            "Eres un analista de sentimientos para operadoras de telecomunicaciones en Perú.\n"
            f"Los siguientes comentarios de usuarios tienen sentimiento '{sentiment}':\n"
            f"{muestra}\n\n"
            "Responde SOLO con un nombre de topic corto (3 a 6 palabras en español) que resuma "
            "el tema principal de estos comentarios. Sin explicaciones, sin comillas, sin puntos."
        )
        response = client.chat.completions.create(
            model=_OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.2,
        )
        name = response.choices[0].message.content.strip().strip('"').strip("'").strip(".")
        return name if name else None
    except Exception as e:
        logger.debug("OpenRouter naming falló: %s", e)
        return None


def _name_cluster_fallback(texts: list[str]) -> str:
    """Nombre offline: los 3 términos más frecuentes del cluster."""
    stopwords = {
        "de", "la", "el", "en", "y", "a", "que", "los", "las", "un", "una",
        "es", "con", "por", "para", "al", "del", "se", "lo", "le", "su",
        "me", "mi", "te", "no", "si", "ya", "pero", "o", "e", "más",
        "muy", "hay", "como", "cuando", "porque", "sin", "todo", "cada",
    }
    counts: Counter = Counter()
    for t in texts:
        for tok in re.findall(r'\b[\w\dáéíóúñü]+\b', t.lower()):
            if tok not in stopwords and len(tok) > 2:
                counts[tok] += 1
    top = [w for w, _ in counts.most_common(4)]
    return " / ".join(top[:3]) if top else "sin topic"


# ─── pipeline principal ───────────────────────────────────────────────────────

def extract_topics(
    comments: list[dict],
    sentiment: Literal["positivo", "negativo", "neutral", "informativo", "irrelevante"],
    min_cluster_size: int = 3,
    max_topics: int = 20,
    use_gemini: bool = True,
) -> dict[str, dict]:
    """
    Extrae topics para un sentimiento dado.
    Devuelve { topic_name: {"frecuencia": int, "representativas": [str]} }
    """
    texts_raw = [
        c.get("comentario", "") or ""
        for c in comments
        if c.get("clasificacion", "").lower() == sentiment
    ]
    texts_clean = [_clean_text(t) for t in texts_raw]
    # filtrar vacíos
    pairs = [(r, c) for r, c in zip(texts_raw, texts_clean) if len(c) > 5]
    if not pairs:
        return {}

    texts_raw_f, texts_clean_f = zip(*pairs)

    labels = _cluster_comments(list(texts_clean_f), min_cluster=min_cluster_size)

    # agrupar por cluster, descartar ruido (-1)
    clusters: dict[int, list[str]] = {}
    for label, raw in zip(labels, texts_raw_f):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(raw)

    if not clusters:
        # todo fue ruido — intentar con threshold más bajo
        clusters = {0: list(texts_raw_f[:50])}

    # ordenar clusters por tamaño descendente, limitar a max_topics
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    sorted_clusters = sorted_clusters[:max_topics]

    result: dict[str, dict] = {}
    seen_names: set[str] = set()

    for cluster_id, cluster_texts in sorted_clusters:
        # naming
        name = None
        if use_gemini:
            name = _name_cluster_llm(cluster_texts, sentiment)
        if not name:
            name = _name_cluster_fallback(cluster_texts)

        # desduplicar nombres
        base_name = name
        suffix = 2
        while name in seen_names:
            name = f"{base_name} ({suffix})"
            suffix += 1
        seen_names.add(name)

        # frases representativas: las más cortas y limpias (más legibles en la nube)
        representativas = sorted(
            set(cluster_texts),
            key=lambda t: len(t),
        )[:3]

        result[name] = {
            "frecuencia": len(cluster_texts),
            "representativas": representativas,
        }

    return result


# ─── generación de JSONs ──────────────────────────────────────────────────────

def generate_topic_json_for_operator(
    operador: str,
    front_dir: Path = None,
    min_cluster_size: int = 3,
    max_topics: int = 20,
    use_gemini: bool = True,
) -> Path | None:
    """
    Genera outputs_front/<operador>/GENERAL_TOPICS.json.
    Devuelve el path escrito o None si no hay datos.
    """
    if front_dir is None:
        front_dir = _ROOT / "outputs_front" / operador
    front_dir.mkdir(parents=True, exist_ok=True)

    comments = _load_classified_comments(operador)
    if not comments:
        logger.warning("Sin comentarios clasificados para %s", operador)
        return None

    output: dict = {
        "operador": operador,
        "total_comentarios": len(comments),
    }
    for sentiment in _SENTIMENTS:
        topics = extract_topics(
            comments,
            sentiment,
            min_cluster_size=min_cluster_size,
            max_topics=max_topics,
            use_gemini=use_gemini,
        )
        if topics:
            output[sentiment] = topics

    out_path = front_dir / "GENERAL_TOPICS.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Topics escritos: %s (%d comentarios)", out_path, len(comments))
    return out_path


def generate_topic_jsons_all_operators(
    operators: list[str] = None,
    **kwargs,
) -> dict[str, Path | None]:
    """Corre generate_topic_json_for_operator para todos los operadores."""
    base_dir = _ROOT / "outputs"
    if operators is None:
        operators = [
            d.name for d in base_dir.iterdir()
            if d.is_dir() and (d / "clasificado").exists()
        ]
    return {op: generate_topic_json_for_operator(op, **kwargs) for op in sorted(operators)}


# ─── figura matplotlib (compatible con el render actual del dashboard) ────────

def build_topic_cloud_figure(
    operador: str,
    sentiment: Literal["positivo", "negativo", "neutral", "informativo", "irrelevante"],
    front_dir: Path = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure | None:
    """
    Carga GENERAL_TOPICS.json y devuelve figura matplotlib con la nube de topics.
    Devuelve None si no hay datos.
    Compatible con st.pyplot() del dashboard.
    """
    if front_dir is None:
        front_dir = _ROOT / "outputs_front" / operador
    topics_path = front_dir / "GENERAL_TOPICS.json"
    if not topics_path.exists():
        return None

    try:
        data = json.loads(topics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    topics = data.get(sentiment, {})
    if not topics:
        return None

    freqs = {name: info["frecuencia"] for name, info in topics.items()}

    from wordcloud import WordCloud
    color = LABEL_COLORS.get(sentiment, "#6B7280")
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        color_func=lambda *_, **__: color,
        stopwords=set(),
        max_words=len(freqs),
        collocations=False,
        prefer_horizontal=0.80,
    ).generate_from_frequencies(freqs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def get_topic_frequencies(
    operador: str,
    sentiment: str,
    front_dir: Path = None,
) -> dict[str, int]:
    """Devuelve {topic_name: frecuencia} desde el JSON generado."""
    if front_dir is None:
        front_dir = _ROOT / "outputs_front" / operador
    topics_path = front_dir / "GENERAL_TOPICS.json"
    if not topics_path.exists():
        return {}
    try:
        data = json.loads(topics_path.read_text(encoding="utf-8"))
        return {
            name: info["frecuencia"]
            for name, info in data.get(sentiment, {}).items()
        }
    except Exception:
        return {}
