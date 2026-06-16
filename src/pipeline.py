"""
Orquestador del pipeline: scrape + clean + classify + save.
Sin dependencias de argparse ni sys.exit — reutilizable desde CLI y dashboard.
"""
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.config_loader import load_config, get_session_path, get_output_dir
from src.utils.io import save_json

logger = logging.getLogger(__name__)


@dataclass
class ScrapeParams:
    platform: str
    username: Optional[str] = None
    keyword: Optional[str] = None
    top_n: int = 5
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    exclude_users: list = field(default_factory=list)
    max_replies: Optional[int] = None
    transcribe: bool = False
    classify: bool = True


@dataclass
class PipelineResult:
    videos: list
    comments: list
    output_dir: Path
    stats: dict


def run_auth(platform: str, username: str = None, config: dict = None) -> None:
    """Guarda sesión de la plataforma indicada. Lanza RuntimeError si falla."""
    if config is None:
        config = load_config()
    session_file = str(get_session_path(platform, config))
    if platform == "tiktok":
        from src.scrapers.tiktok.auth import save_session
        save_session(username=username, session_file=session_file, config=config)
    else:
        from src.scrapers.facebook.auth import save_session
        save_session(session_file=session_file, config=config)


def run_scrape(params: ScrapeParams, config: dict = None) -> PipelineResult:
    """
    Ejecuta scraping + limpieza + clasificación opcional.
    Lanza RuntimeError si la sesión no es válida.
    """
    if config is None:
        config = load_config()

    from src.preprocessing.cleaner import agregar_columna_departamento

    session_file = str(get_session_path(params.platform, config))
    output_dir = get_output_dir(params.platform, config)

    if params.platform == "tiktok":
        results, base_name = _scrape_tiktok(params, session_file, config)
        text_field = "comentario"
    else:
        results, base_name = _scrape_facebook(params, session_file, config)
        text_field = "texto"

    from src.preprocessing.cleaner import clean_comments

    all_videos = _extract_videos(results, params.platform)
    all_comments_raw = _flatten_comments(results, params.platform)
    all_comments_with_replies = _flatten_replies(all_comments_raw, text_field)
    all_comments_cleaned = clean_comments(all_comments_with_replies, config, text_field=text_field)
    all_comments = agregar_columna_departamento(all_comments_cleaned, text_field=text_field)

    save_json(all_videos, output_dir, "videos.json")
    save_json(all_comments, output_dir, "comments.json")

    if params.transcribe and params.platform == "tiktok":
        from src.transcription.transcriber import batch_transcribe
        transcripts = batch_transcribe(all_videos, config, output_dir)
        save_json(transcripts, output_dir, "transcripts.json")

    if params.classify:
        all_comments = run_classify(
            all_comments, config, output_dir,
            platform=params.platform, text_field=text_field,
        )

    return PipelineResult(
        videos=all_videos,
        comments=all_comments,
        output_dir=output_dir,
        stats=_build_stats(all_comments),
    )


def run_classify(
    comments: list,
    config: dict = None,
    output_dir: Path = None,
    platform: str = "tiktok",
    text_field: str = "comentario",
) -> list:
    """Clasifica una lista de comentarios y los guarda si se indica output_dir."""
    if config is None:
        config = load_config()
    from src.retrieval.classifier import CommentClassifier
    clf = CommentClassifier(config)
    classified = clf.classify_comments(comments, text_field=text_field, platform=platform)
    if output_dir:
        save_json(classified, output_dir, "comments.json")
        clf.save_embeddings(classified, output_dir / "embeddings", text_field=text_field)
    return classified


def run_classify_from_files(
    operador: str,
    config: dict = None,
    base_dir: Path = None,
) -> dict:
    """
    Lee todos los archivos preprocesados de TikTok y Facebook de
    outputs/<operador>/preprocesado/ y clasifica sus comentarios.

    TikTok:   VIDEO_CLEANER_*.json  → clasificado/VIDEO_CLASSIFIED_*.json
    Facebook: facebook_clean_*.json → clasificado/facebook_classified_*.json

    Devuelve un resumen con total de archivos procesados y conteo de labels.
    """
    if config is None:
        config = load_config()
    if base_dir is None:
        base_dir = Path("outputs")

    input_dir = base_dir / operador / "preprocesado"
    output_dir = base_dir / operador / "clasificado"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Definición de cada plataforma: (glob, id_field, out_prefix, plataforma)
    _PLATFORMS = [
        ("VIDEO_CLEANER_*.json",  "video_id",  "VIDEO_CLEANER_",   "VIDEO_CLASSIFIED_",   "tiktok"),
        ("facebook_clean_*.json", "post_id",   "facebook_clean_",  "facebook_classified_", "facebook"),
    ]

    from src.retrieval.classifier import CommentClassifier
    from src.preprocessing.cleaner import agregar_columna_departamento
    clf = CommentClassifier(config)

    total_archivos = 0
    total_comentarios = 0
    total_counts: Counter = Counter()

    for glob_pat, id_field, in_prefix, out_prefix, platform in _PLATFORMS:
        archivos = sorted(input_dir.glob(glob_pat))
        if not archivos:
            logger.info("Sin archivos %s para %s/%s", glob_pat, operador, platform)
            continue

        for archivo in archivos:
            try:
                with open(archivo, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Error leyendo %s: %s", archivo.name, e)
                continue

            item_id = data.get(id_field, archivo.stem.replace(in_prefix, ""))
            comentarios = data.get("comentarios", [])

            if not comentarios:
                continue

            classified = clf.classify_comments(
                comentarios, text_field="comentario", platform=platform
            )
            classified = agregar_columna_departamento(classified, text_field="comentario")

            resultado = {
                id_field:            item_id,
                "operador":          data.get("operador", operador),
                "plataforma":        platform,
                "total_comentarios": len(classified),
                "comentarios":       classified,
            }
            # Campos opcionales según plataforma
            if platform == "tiktok":
                resultado["fecha_video"] = data.get("fecha_video", "")
            else:
                resultado["post_url"]   = data.get("post_url", "")
                resultado["fecha_post"] = data.get("fecha_post", "")

            out_file = output_dir / f"{out_prefix}{item_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)

            counts = Counter(c.get("clasificacion", "sin_clasificar") for c in classified)
            total_counts += counts
            total_comentarios += len(classified)
            total_archivos += 1
            logger.info("[%s] Clasificado %s → %s", platform, archivo.name, dict(counts))

    resumen = {
        "operador":   operador,
        "archivos":   total_archivos,
        "comentarios": total_comentarios,
        **dict(total_counts),
    }
    logger.info("Resumen %s: %s", operador, resumen)

    # Generar phrase clouds para el dashboard
    from src.word_clouds.phrase_cloud import generate_phrase_jsons_for_operator
    generate_phrase_jsons_for_operator(operador)

    return resumen


def run_clean_operator(operador: str, config: dict = None, base_dir: Path = None) -> int:
    """
    Lee los JSONs crudos de outputs/<op>/mes_*/ (video_<id>.json),
    limpia y preprocesa los comentarios de cada video y guarda en
    outputs/<op>/preprocesado/VIDEO_CLEANER_<id>.json.
    Devuelve cantidad de videos procesados.
    """
    if config is None:
        config = load_config()
    if base_dir is None:
        base_dir = Path("outputs")

    from src.preprocessing.cleaner import clean_comments, agregar_columna_departamento

    op_dir = base_dir / operador
    output_dir = op_dir / "preprocesado"
    output_dir.mkdir(parents=True, exist_ok=True)

    # busca en todas las subcarpetas mes_* y directamente en op_dir
    raw_files = sorted(op_dir.rglob("video_*.json"))
    if not raw_files:
        logger.warning("No se encontraron JSONs crudos en %s", op_dir)
        return 0

    procesados = 0
    for archivo in raw_files:
        try:
            with open(archivo, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Error leyendo %s: %s", archivo.name, e)
            continue

        video_id = data.get("video", {}).get("id") or archivo.stem.replace("video_", "")
        comentarios_raw = data.get("comentarios", [])
        if not comentarios_raw:
            continue

        comentarios_limpios = clean_comments(comentarios_raw, config, text_field="comentario")

        resultado = {
            "video_id": video_id,
            "operador": operador,
            "fecha_video": data.get("fecha_video", ""),
            "total_comentarios": len(comentarios_limpios),
            "comentarios": comentarios_limpios,
        }

        out_file = output_dir / f"VIDEO_CLEANER_{video_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)

        logger.info("Limpiado %s → %d comentarios", archivo.name, len(comentarios_limpios))
        procesados += 1

    return procesados


def run_process_all(base_dir: Path = None, config: dict = None) -> list[dict]:
    """
    Para cada operador detectado en outputs/:
      1. clean    → outputs/<op>/preprocesado/VIDEO_CLEANER_*.json
      2. classify → outputs/<op>/clasificado/VIDEO_CLASSIFIED_*.json
      3. phrases  → outputs_front/<op>/VIDEO_PHRASES_*.json
    Devuelve lista de resúmenes por operador.
    """
    if base_dir is None:
        base_dir = Path("outputs")
    if config is None:
        config = load_config()

    _SKIP = {"output_tiktok", "output_facebook", "listas"}
    operators = sorted(
        d.name for d in base_dir.iterdir()
        if d.is_dir() and d.name not in _SKIP
        and not d.name.endswith(".pkl")
    )

    if not operators:
        logger.warning("No se encontraron operadores en %s", base_dir)
        return []

    resumenes = []
    for op in operators:
        logger.info("── %s ──", op.upper())
        videos_limpios = run_clean_operator(op, config=config, base_dir=base_dir)
        logger.info("  Limpieza: %d videos procesados", videos_limpios)
        resumen = run_classify_from_files(op, config=config, base_dir=base_dir)
        resumenes.append(resumen)

    return resumenes


def run_search(
    query: str,
    platform: str,
    top_k: int = 10,
    config: dict = None,
) -> list:
    """Busca comentarios similares a query usando similitud coseno."""
    if config is None:
        config = load_config()
    from src.retrieval.classifier import CommentClassifier
    clf = CommentClassifier(config)
    return clf.search_similar(query=query, platform=platform, top_k=top_k)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _scrape_tiktok(params: ScrapeParams, session_file: str, config: dict):
    from src.scrapers.tiktok.auth import check_session
    from src.scrapers.tiktok import scraper as tt
    from src.scrapers.tiktok import keyword_scraper as tt_kw

    if not check_session(session_file):
        raise RuntimeError("Sesión de TikTok inválida o expirada")

    if params.keyword:
        results = tt_kw.search_by_keyword(
            keyword=params.keyword,
            top_n=params.top_n,
            session_file=session_file,
            config=config,
            date_from=params.date_from,
            date_to=params.date_to,
            exclude_users=params.exclude_users,
            max_replies=params.max_replies,
        )
        return results, params.keyword.replace(" ", "_")

    results = tt.search_by_user(
        username=params.username,
        top_n=params.top_n,
        session_file=session_file,
        config=config,
        date_from=params.date_from,
        date_to=params.date_to,
        exclude_users=params.exclude_users,
        max_replies=params.max_replies,
    )
    return results, params.username


def _scrape_facebook(params: ScrapeParams, session_file: str, config: dict):
    from src.scrapers.facebook.auth import check_session
    from src.scrapers.facebook import scraper as fb

    if not check_session(session_file):
        raise RuntimeError("Sesión de Facebook inválida o expirada")

    if params.keyword:
        results = fb.search_by_keyword(
            keyword=params.keyword,
            top_n=params.top_n,
            session_file=session_file,
            config=config,
            date_from=params.date_from,
            date_to=params.date_to,
        )
        return results, params.keyword.replace(" ", "_")

    results = fb.search_by_user(
        username=params.username,
        top_n=params.top_n,
        session_file=session_file,
        config=config,
        date_from=params.date_from,
        date_to=params.date_to,
    )
    return results, params.username


def _extract_videos(results: list, platform: str) -> list:
    key = "video" if platform == "tiktok" else "post"
    return [r[key] for r in results if r and key in r]


def _flatten_comments(results: list, platform: str) -> list:
    out = []
    is_tiktok = platform == "tiktok"
    parent_key = "video" if is_tiktok else "post"
    id_field = "video_id" if is_tiktok else "post_url"
    id_source = "id" if is_tiktok else "url"

    for r in results:
        if not r:
            continue
        parent_val = r[parent_key].get(id_source, "")
        for c in r["comments"]:
            c[id_field] = parent_val
            out.append(c)
    return out


def _flatten_replies(comments: list, text_field: str = "comentario") -> list:
    """
    Expande las replies anidadas como entradas de primer nivel.
    Cada reply hereda video_id/post_url y recibe parent_id apuntando al comentario padre.
    El comentario padre conserva su lista 'respuestas' intacta (para referencia),
    pero las replies también quedan disponibles como comentarios independientes
    para clasificación y búsqueda semántica.
    """
    out = []
    replies_field = "respuestas"

    for comment in comments:
        out.append(comment)
        parent_id = comment.get("id", "")
        video_id = comment.get("video_id") or comment.get("post_url", "")

        for reply in comment.get(replies_field, []):
            flat_reply = dict(reply)
            flat_reply["parent_id"] = parent_id
            flat_reply["video_id"] = video_id
            flat_reply["es_reply"] = True
            # Normalizar el campo de texto al mismo nombre que el padre
            if text_field not in flat_reply and "comentario" not in flat_reply:
                flat_reply[text_field] = flat_reply.get("comentario", "")
            out.append(flat_reply)

    return out


def _build_stats(comments: list) -> dict:
    labels = [c.get("clasificacion", "sin_clasificar") for c in comments]
    counts = dict(Counter(labels))
    return {"total": len(comments), **counts}
