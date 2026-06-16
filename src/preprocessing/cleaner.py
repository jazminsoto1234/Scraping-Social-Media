"""
Limpieza y normalizacion de comentarios extraidos.
Filtra spam, emojis vacios, comentarios irrelevantes y duplicados.
"""
import re
import unicodedata
import logging
import json
import argparse
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_SPAM_PATTERNS = [
    r"^(jaja|jeje|lol|haha|xd|hehe|jajaja|jajajaja)+[!?.\s]*$",
    r"^(primero|first|1ro|1st|segundo|2do)\b[!?.\s]*$",
    r"^[.,:;\-_\s]+$",
    r"^\d+$",
    r"^(si|no|ok|oke|dale|va|claro)\b[!?.\s]*$",
    r"^(❤|👍|👏|😂|🔥|💯|✅|🙌)+$",
]

_COMPILED_SPAM = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _SPAM_PATTERNS]


def _serialize_comment(comment: dict) -> str:
    """Serializa un comentario completo para comparación de duplicados."""
    # Crear una versión serializable del comentario (excluyendo respuestas/replies)
    comment_copy = dict(comment)
    comment_copy.pop("respuestas", None)
    comment_copy.pop("replies", None)
    return json.dumps(comment_copy, sort_keys=True, ensure_ascii=False)


def clean_comments(
    comments: list[dict],
    config: dict,
    text_field: str = "comentario",
    exclude_users: list[str] | None = None,
) -> list[dict]:
    """
    Filtra y limpia una lista de comentarios, eliminando duplicados por video.
    Los duplicados se detectan solo dentro del mismo video/post.
    Se considera duplicado si el comentario COMPLETO es idéntico (todos los campos).

    Args:
        comments: lista de dicts con al menos el campo text_field.
        config: configuracion del proyecto (lee preprocessing.*).
        text_field: nombre del campo que contiene el texto (default: 'comentario').
        exclude_users: lista de usuario_id a excluir (siempre excluye comentarios del usuario scrapeado).

    Returns:
        Lista de comentarios limpios con sus respuestas tambien filtradas.
    """
    cfg = config.get("preprocessing", {})
    min_len = cfg.get("min_comment_length", 3)
    emoji_threshold = cfg.get("emoji_only_threshold", 0.8)
    excluded_users = {u.lower() for u in (exclude_users or [])}

    def _is_user_excluded(user_id: str) -> bool:
        return user_id.lower() in excluded_users if excluded_users else False

    cleaned = []
    removed = 0
    excluded_by_user = 0
    # Duplicados por video: {video_id: set(comentarios_completos_serializados)}
    seen_comments_by_video = {}

    for comment in comments:
        user_id = comment.get("usuario_id", "")
        if _is_user_excluded(user_id):
            excluded_by_user += 1
            continue

        text = comment.get(text_field, "") or ""
        text_clean = normalize_text(text)

        if not _is_valid(text_clean, min_len, emoji_threshold):
            removed += 1
            continue

        # Obtener ID del video/post
        video_id = comment.get("video_id") or comment.get("post_url") or "unknown"

        if video_id not in seen_comments_by_video:
            seen_comments_by_video[video_id] = set()

        # Serializar comentario completo para comparación
        comment_key = _serialize_comment(comment)
        if comment_key in seen_comments_by_video[video_id]:
            removed += 1
            continue

        seen_comments_by_video[video_id].add(comment_key)

        # Limpiar respuestas del mismo comentario (duplicados globales en replies)
        replies_field = "respuestas" if "respuestas" in comment else "replies"
        replies = comment.get(replies_field, [])
        cleaned_replies = []
        seen_reply_keys = set()

        for reply in replies:
            reply_user_id = reply.get("usuario_id", "")
            if _is_user_excluded(reply_user_id):
                continue

            r_text = normalize_text(reply.get(text_field, "") or "")
            if _is_valid(r_text, min_len, emoji_threshold):
                reply_key = _serialize_comment(reply)
                if reply_key not in seen_reply_keys:
                    seen_reply_keys.add(reply_key)
                    cleaned_replies.append(reply)

        new_comment = dict(comment)
        new_comment[replies_field] = cleaned_replies
        cleaned.append(new_comment)

    logger.info(
        "Comentarios originales: %d | Excluidos por usuario: %d | Removidos (spam+duplicados): %d | Resultado: %d",
        len(comments), excluded_by_user, removed, len(cleaned)
    )
    return cleaned


def normalize_text(text: str) -> str:
    """Normaliza el texto: quita espacios extra, normaliza unicode, convierte a minúsculas."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_emoji_heavy(text: str, threshold: float = 0.8) -> bool:
    """Check if text exceeds emoji threshold, excluding variation selectors."""
    if not text:
        return True
    visible = [c for c in text if unicodedata.category(c) not in ("Mn", "Cf")]
    if not visible:
        return True
    emoji_count = sum(1 for c in visible if unicodedata.category(c) in ("So", "Sm"))
    return emoji_count / len(visible) >= threshold


def _is_valid(text: str, min_len: int, emoji_threshold: float = 0.8) -> bool:
    """Valida que el comentario tenga contenido semantico."""
    if not text or len(text) < min_len:
        return False
    if is_emoji_heavy(text, emoji_threshold):
        return False
    for pattern in _COMPILED_SPAM:
        if pattern.match(text):
            return False

    # Detectar ruido aleatorio (secuencias sin sentido)
    if _is_random_noise(text):
        return False

    return True

def _is_random_noise(text: str) -> bool:
    """Detecta comentarios que son puramente ruido/spam aleatorio.

    Ejemplos:
    - 'dghdgbcgbcfhczfghkiedtulbfurfjhkkhd cnldgkjfhjfhiikdldkfjfjfkkfkdkfkjfjffjfjjdkdjfjfjffj'
    - Secuencias de caracteres sin consonantes-vocales normales
    - Palabras inventadas repetidas
    """
    # Buscar secuencias largas (>15 chars) de caracteres consonantes sin vocales
    long_consonant_seqs = re.findall(r'[bcdfghjklmnpqrstvwxyz]{15,}', text.lower())
    if long_consonant_seqs:
        return True

    # Si tiene palabras muy largas (>20 chars) sin estructura de palabra normal
    words = text.split()
    for word in words:
        if len(word) > 20:
            # Palabras normales tienen vocales, palabras inventadas no
            vowels = sum(1 for c in word.lower() if c in 'aeiouáéíóú')
            if vowels == 0:  # Sin vocales = ruido
                return True

    return False


_DEPTO_CANONICAL: dict[str, str] = {
    "amazonas":    "Amazonas",
    "ancash":      "Áncash",
    "apurimac":    "Apurímac",
    "arequipa":    "Arequipa",
    "ayacucho":    "Ayacucho",
    "cajamarca":   "Cajamarca",
    "callao":      "Callao",
    "cusco":       "Cusco",
    "cuzco":       "Cusco",
    "huancavelica":"Huancavelica",
    "huanuco":     "Huánuco",
    "ica":         "Ica",
    "junin":       "Junín",
    "la libertad": "La Libertad",
    "lambayeque":  "Lambayeque",
    "lima":        "Lima",
    "loreto":      "Loreto",
    "madre de dios":"Madre de Dios",
    "moquegua":    "Moquegua",
    "pasco":       "Pasco",
    "piura":       "Piura",
    "puno":        "Puno",
    "san martin":  "San Martín",
    "tacna":       "Tacna",
    "tumbes":      "Tumbes",
    "ucayali":     "Ucayali",
}

_GENTILICIO_MAP: dict[str, str] = {
    "amazonense": "amazonas", "chachapoyas": "amazonas",
    "ancashino": "ancash", "ancashina": "ancash",
    "huaracino": "ancash", "huaracina": "ancash",
    "chimbotano": "ancash", "chimbotana": "ancash",
    "apurimaqueno": "apurimac", "apurimaquena": "apurimac",
    "abancaino": "apurimac", "abancaina": "apurimac",
    "arequipeno": "arequipa", "arequipena": "arequipa",
    "misti": "arequipa",
    "ayacuchano": "ayacucho", "ayacuchana": "ayacucho",
    "huamanguino": "ayacucho", "huamanguina": "ayacucho",
    "cajamarquino": "cajamarca", "cajamarquina": "cajamarca",
    "chalaco": "callao", "chalaca": "callao",
    "cusqueno": "cusco", "cusquena": "cusco",
    "cuzqueno": "cusco", "cuzquena": "cusco",
    "cusceno": "cusco",
    "huancavelicano": "huancavelica", "huancavelicana": "huancavelica",
    "huanuqueno": "huanuco", "huanuquena": "huanuco",
    "icano": "ica", "icana": "ica", "iqueño": "ica", "iqueña": "ica",
    "iceno": "ica",
    "junino": "junin", "junina": "junin",
    "huancaino": "junin", "huancaina": "junin",
    "liberteño": "la libertad", "liberteña": "la libertad",
    "liberteno": "la libertad", "libertena": "la libertad",
    "trujillano": "la libertad", "trujillana": "la libertad",
    "lambayecano": "lambayeque", "lambayecana": "lambayeque",
    "chiclayano": "lambayeque", "chiclayanas": "lambayeque",
    "limeno": "lima", "limena": "lima",
    "limeño": "lima", "limeña": "lima",
    "capitalino": "lima", "capitalina": "lima",
    "loretano": "loreto", "loretana": "loreto",
    "iquiteño": "loreto", "iquiteña": "loreto",
    "iquiteno": "loreto", "iquitena": "loreto",
    "madre dios": "madre de dios",
    "madredioseño": "madre de dios", "madredioseña": "madre de dios",
    "moqueguano": "moquegua", "moqueguana": "moquegua",
    "pasqueno": "pasco", "pasquena": "pasco",
    "cerreño": "pasco",
    "piurano": "piura", "piurana": "piura",
    "puneno": "puno", "punena": "puno",
    "puneño": "puno", "puneña": "puno",
    "juliaqueno": "puno",
    "sanmartinense": "san martin",
    "moyobambino": "san martin",
    "tarapotino": "san martin",
    "tacneño": "tacna", "tacneña": "tacna",
    "tacneno": "tacna", "tacnena": "tacna",
    "tumbesino": "tumbes", "tumbesina": "tumbes",
    "ucayalino": "ucayali", "ucayalina": "ucayali",
    "pucallpino": "ucayali", "pucallpina": "ucayali",
}

_CIUDAD_MAP: dict[str, str] = {
    "lima":         "lima",
    "arequipa":     "arequipa",
    "trujillo":     "la libertad",
    "chiclayo":     "lambayeque",
    "piura":        "piura",
    "iquitos":      "loreto",
    "cusco":        "cusco",
    "cuzco":        "cusco",
    "huancayo":     "junin",
    "tacna":        "tacna",
    "pucallpa":     "ucayali",
    "chimbote":     "ancash",
    "huaraz":       "ancash",
    "ica":          "ica",
    "juliaca":      "puno",
    "puno":         "puno",
    "tumbes":       "tumbes",
    "moquegua":     "moquegua",
    "huanuco":      "huanuco",
    "cajamarca":    "cajamarca",
    "ayacucho":     "ayacucho",
    "huancavelica": "huancavelica",
    "callao":       "callao",
    "abancay":      "apurimac",
    "puerto maldonado": "madre de dios",
    "moyobamba":    "san martin",
    "tarapoto":     "san martin",
    "cerro de pasco": "pasco",
}

_TYPOS_MAP: dict[str, str] = {
    "arequipa":   "arequipa",
    "areqipa":    "arequipa",
    "arequia":    "arequipa",
    "cusco":      "cusco",
    "qosco":      "cusco",
    "cuzco":      "cusco",
    "anqash":     "ancash",
    "ancahs":     "ancash",
    "piurra":     "piura",
    "trujilo":    "la libertad",
    "chiklayo":   "lambayeque",
    "chiclallo":  "lambayeque",
    "iquito":     "loreto",
    "apurimac":   "apurimac",
    "apurimack":  "apurimac",
    "cajamrca":   "cajamarca",
}


def _strip_accents(s: str) -> str:
    """Remove accents and convert to lowercase."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def extraer_departamento(texto: str) -> str:
    """
    Extract Peruvian department from text.

    Searches for official names, cities, gentilics, and typos.
    Returns the earliest match (longest pattern in case of ties) or "No especificado".
    """
    if not texto or not texto.strip():
        return "No especificado"

    normalized = _strip_accents(texto)
    candidates: list[tuple[int, int, str]] = []

    all_names: dict[str, str] = {**_TYPOS_MAP}
    for k in _DEPTO_CANONICAL:
        all_names[k] = k
    for key, canonical_key in all_names.items():
        m = re.search(r'\b' + re.escape(key) + r'\b', normalized)
        if m:
            name = _DEPTO_CANONICAL.get(canonical_key, canonical_key.title())
            candidates.append((m.start(), -len(key), name))

    for city, canonical_key in _CIUDAD_MAP.items():
        m = re.search(r'\b' + re.escape(_strip_accents(city)) + r'\b', normalized)
        if m:
            name = _DEPTO_CANONICAL.get(canonical_key, canonical_key.title())
            candidates.append((m.start(), -len(city), name))

    for gent, canonical_key in _GENTILICIO_MAP.items():
        m = re.search(r'\b' + re.escape(_strip_accents(gent)) + r'\b', normalized)
        if m:
            name = _DEPTO_CANONICAL.get(canonical_key, canonical_key.title())
            candidates.append((m.start(), -len(gent), name))

    if not candidates:
        return "No especificado"

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def agregar_columna_departamento(
    comments: list[dict],
    text_field: str = "comentario",
) -> list[dict]:
    """Add 'departamento' field to each comment using department extraction."""
    result = []
    for c in comments:
        item = dict(c)
        item["departamento"] = extraer_departamento(item.get(text_field, "") or "")
        result.append(item)
    return result


def detect_language(text: str) -> str:
    """Simple language detection based on character set (es/en/other)."""
    spanish_chars = set("áéíóúüñ¿¡")
    has_spanish = any(c in spanish_chars for c in text.lower())
    if has_spanish:
        return "es"
    latin = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    if latin > 0:
        return "en"
    return "other"


# ---------------------------------------------------------------------------
# Pipeline de operadores: flatten → dedup semántico → clean → guardar
# ---------------------------------------------------------------------------

def _normalize_for_dedup(text: str) -> str:
    """Normalización agresiva para comparación de duplicados semánticos.
    Va más allá de normalize_text(): elimina puntuación y caracteres especiales,
    colapsa espacios — captura 'Pésimo!!' == 'pesimo', 'qué malo?' == 'que malo'.
    """
    text = unicodedata.normalize("NFD", text.lower())
    # quitar diacríticos
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # quitar todo lo que no sea letra, número o espacio
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    # quitar dígitos solos (números sin contexto)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def flatten_video(raw: dict) -> list[dict]:
    """Aplana un video_{ID}.json en lista plana de comentarios + replies.

    Descarta: usuario_id, nickname, likes, num_respuestas.
    Cada entrada conserva solo: id, comentario, fecha, es_reply, parent_id, video_id.
    """
    video_id = raw.get("video", {}).get("id", "unknown")
    result = []

    for comment in raw.get("comentarios", []):
        cid = comment.get("id", "")
        result.append({
            "id":        cid,
            "comentario": comment.get("comentario", "") or "",
            "fecha":     comment.get("fecha", ""),
            "es_reply":  False,
            "parent_id": None,
            "video_id":  video_id,
        })
        for reply in comment.get("respuestas", []):
            result.append({
                "id":        reply.get("id", ""),
                "comentario": reply.get("comentario", "") or "",
                "fecha":     reply.get("fecha", ""),
                "es_reply":  True,
                "parent_id": cid,
                "video_id":  video_id,
            })

    return result


def dedup_semantic(comments: list[dict], text_field: str = "comentario") -> list[dict]:
    """Elimina comentarios cuyo texto es 'básicamente el mismo' dentro del mismo video.

    Dos comentarios se consideran duplicados si su texto normalizado agresivamente
    (_normalize_for_dedup) es idéntico. Conserva la primera aparición.
    """
    seen: dict[str, set] = {}  # video_id → set de textos normalizados
    result = []

    for c in comments:
        vid = c.get("video_id", "unknown")
        key = _normalize_for_dedup(c.get(text_field, "") or "")

        if not key:
            continue

        if vid not in seen:
            seen[vid] = set()

        if key in seen[vid]:
            continue

        seen[vid].add(key)
        result.append(c)

    return result


def clean_operator_videos(
    operador: str,
    periodo: str,
    base_dir: Path = Path("outputs"),
) -> None:
    """Lee todos los video_*.json de un operador/periodo, aplica el pipeline de
    limpieza y guarda en outputs/{operador}/preprocesado/VIDEO_CLEANER_{ID}.json.

    Pipeline:
      1. flatten_video     — aplana comentarios + replies, descarta usuario/likes
      2. dedup_semantic    — elimina duplicados semánticos por video
      3. _is_valid         — filtra spam, emojis, ruido aleatorio
    Sin clasificación.
    """
    cfg_preprocessing = {"min_comment_length": 3, "emoji_only_threshold": 0.8}
    min_len = cfg_preprocessing["min_comment_length"]
    emoji_threshold = cfg_preprocessing["emoji_only_threshold"]

    input_dir = base_dir / operador / f"mes_{periodo}"
    output_dir = base_dir / operador / "preprocesado"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob("video_*.json"))
    if not video_files:
        logger.warning("No se encontraron archivos en %s", input_dir)
        return

    total_raw = 0
    total_out = 0

    for vf in video_files:
        try:
            raw = json.loads(vf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Error leyendo %s: %s", vf, e)
            continue

        video_id = raw.get("video", {}).get("id", vf.stem.replace("video_", ""))

        flat = flatten_video(raw)
        deduped = dedup_semantic(flat)

        cleaned = []
        for c in deduped:
            text_norm = normalize_text(c.get("comentario", "") or "")
            if _is_valid(text_norm, min_len, emoji_threshold):
                cleaned.append(c)

        total_raw += len(flat)
        total_out += len(cleaned)

        out_data = {
            "video_id":          video_id,
            "operador":          raw.get("operador", operador),
            "fecha_video":       raw.get("fecha_video", ""),
            "total_comentarios": len(cleaned),
            "comentarios":       cleaned,
        }

        out_path = output_dir / f"VIDEO_CLEANER_{video_id}.json"
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "%s → %d raw, %d después de dedup+clean → %s",
            vf.name, len(flat), len(cleaned), out_path.name,
        )

    print(
        f"[cleaner] {operador}/{periodo}: {len(video_files)} videos | "
        f"{total_raw} comentarios raw → {total_out} limpios"
    )
    print(f"[cleaner] Guardado en: {output_dir}")


# ---------------------------------------------------------------------------
# Pipeline Facebook: flatten → dedup semántico → clean → guardar
# ---------------------------------------------------------------------------

def flatten_post(raw: dict) -> list[dict]:
    """Aplana un post_{ID}.json de Facebook en lista plana de comentarios + replies.

    Cada entrada conserva: id, comentario, fecha, es_reply, parent_id, post_id.
    El campo de texto origen es 'texto', normalizado a 'comentario' para
    compartir el mismo pipeline de limpieza que TikTok.
    """
    post_url = raw.get("post", {}).get("url", "")
    post_id = post_url.rstrip("/").split("/")[-1] if post_url else "unknown"
    result = []

    for comment in raw.get("comentarios", []):
        cid = comment.get("id", "")
        result.append({
            "id":         cid,
            "comentario": comment.get("texto", "") or "",
            "fecha":      comment.get("fecha", ""),
            "es_reply":   False,
            "parent_id":  None,
            "post_id":    post_id,
        })
        for reply in comment.get("respuestas", []):
            result.append({
                "id":         reply.get("id", ""),
                "comentario": reply.get("texto", "") or "",
                "fecha":      reply.get("fecha", ""),
                "es_reply":   True,
                "parent_id":  cid,
                "post_id":    post_id,
            })

    return result


def dedup_semantic_facebook(comments: list[dict], text_field: str = "comentario") -> list[dict]:
    """Elimina duplicados semánticos dentro del mismo post de Facebook."""
    seen: dict[str, set] = {}
    result = []

    for c in comments:
        pid = c.get("post_id", "unknown")
        key = _normalize_for_dedup(c.get(text_field, "") or "")

        if not key:
            continue

        if pid not in seen:
            seen[pid] = set()

        if key in seen[pid]:
            continue

        seen[pid].add(key)
        result.append(c)

    return result


def clean_operator_facebook(
    operador: str,
    periodo: str,
    base_dir: Path = Path("outputs"),
) -> None:
    """Lee todos los post_*.json de outputs/{operador}/mes_{periodo}/facebook/,
    aplica el pipeline de limpieza y guarda en outputs/{operador}/preprocesado/
    como facebook_clean_{ID}.json.

    Pipeline:
      1. flatten_post      — aplana comentarios + replies, mapea 'texto' → 'comentario'
      2. dedup_semantic    — elimina duplicados semánticos por post
      3. _is_valid         — filtra spam, emojis, ruido aleatorio
    """
    cfg_preprocessing = {"min_comment_length": 3, "emoji_only_threshold": 0.8}
    min_len = cfg_preprocessing["min_comment_length"]
    emoji_threshold = cfg_preprocessing["emoji_only_threshold"]

    input_dir = base_dir / operador / f"mes_{periodo}" / "facebook"
    output_dir = base_dir / operador / "preprocesado"
    output_dir.mkdir(parents=True, exist_ok=True)

    post_files = sorted(input_dir.glob("post_*.json"))
    if not post_files:
        logger.warning("No se encontraron archivos en %s", input_dir)
        return

    total_raw = 0
    total_out = 0

    for pf in post_files:
        try:
            raw = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Error leyendo %s: %s", pf, e)
            continue

        post_url = raw.get("post", {}).get("url", "")
        post_id = pf.stem.replace("post_", "")

        flat = flatten_post(raw)
        deduped = dedup_semantic_facebook(flat)

        cleaned = []
        for c in deduped:
            text_norm = normalize_text(c.get("comentario", "") or "")
            if _is_valid(text_norm, min_len, emoji_threshold):
                cleaned.append(c)

        total_raw += len(flat)
        total_out += len(cleaned)

        out_data = {
            "post_id":           post_id,
            "post_url":          post_url,
            "operador":          raw.get("operador", operador),
            "fecha_post":        raw.get("post", {}).get("fecha", ""),
            "total_comentarios": len(cleaned),
            "comentarios":       cleaned,
        }

        out_path = output_dir / f"facebook_clean_{post_id}.json"
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "%s → %d raw, %d después de dedup+clean → %s",
            pf.name, len(flat), len(cleaned), out_path.name,
        )

    print(
        f"[cleaner:facebook] {operador}/{periodo}: {len(post_files)} posts | "
        f"{total_raw} comentarios raw → {total_out} limpios"
    )
    print(f"[cleaner:facebook] Guardado en: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Preprocesa comentarios de TikTok y Facebook.\n"
            "  TikTok  : outputs/{op}/mes_{periodo}/video_*.json       → preprocesado/VIDEO_CLEANER_{ID}.json\n"
            "  Facebook: outputs/{op}/mes_{periodo}/facebook/post_*.json → preprocesado/facebook_clean_{ID}.json\n\n"
            "Ejemplos:\n"
            "  python -m src.preprocessing.cleaner --operador movistar --periodo 202605\n"
            "  python -m src.preprocessing.cleaner --operadores movistar entel claro bitel --periodo 202605\n"
            "  python -m src.preprocessing.cleaner --operadores movistar --all-periodos\n"
            "  python -m src.preprocessing.cleaner --all-operadores --all-periodos\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    op_group = parser.add_mutually_exclusive_group(required=True)
    op_group.add_argument("--operador",       metavar="OP",  help="Un operador (ej: movistar)")
    op_group.add_argument("--operadores",     nargs="+", metavar="OP", help="Uno o más operadores")
    op_group.add_argument("--all-operadores", action="store_true", help="Todos los operadores en base-dir")

    per_group = parser.add_mutually_exclusive_group(required=True)
    per_group.add_argument("--periodo",      metavar="YYYYMM", help="Período específico (ej: 202605)")
    per_group.add_argument("--all-periodos", action="store_true", help="Todos los mes_* encontrados por operador")

    parser.add_argument("--base-dir", default="outputs", metavar="DIR",
                        help="Directorio base de outputs (default: outputs)")

    args = parser.parse_args()
    base = Path(args.base_dir)

    # Resolver lista de operadores
    _SKIP = {"output_tiktok", "output_facebook", "listas", "preprocesado", "clasificado"}
    if args.all_operadores:
        operadores = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and d.name not in _SKIP and not d.name.endswith(".pkl")
        )
    elif args.operadores:
        operadores = args.operadores
    else:
        operadores = [args.operador]

    for operador in operadores:
        op_dir = base / operador
        if not op_dir.exists():
            print(f"[cleaner] Carpeta no encontrada: {op_dir}, saltando")
            continue

        # Resolver periodos
        if args.all_periodos:
            periodos = sorted(
                d.name.replace("mes_", "")
                for d in op_dir.iterdir()
                if d.is_dir() and d.name.startswith("mes_")
            )
            if not periodos:
                print(f"[cleaner] {operador}: sin carpetas mes_*, saltando")
                continue
        else:
            periodos = [args.periodo]

        for periodo in periodos:
            clean_operator_videos(operador=operador, periodo=periodo, base_dir=base)
            clean_operator_facebook(operador=operador, periodo=periodo, base_dir=base)
