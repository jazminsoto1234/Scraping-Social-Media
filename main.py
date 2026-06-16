"""
Pipeline principal: scraping + preprocesamiento + transcripcion + clasificacion.

Uso:
  # Autenticacion (una sola vez por plataforma)
  python main.py tiktok auth --username entel_peru
  python main.py facebook auth

  # Scraping por usuario
  python main.py tiktok scrape --username entel_peru --top-n 5
  python main.py facebook scrape --username EntelPeru --top-n 3

  # Scraping por keyword
  python main.py tiktok scrape --keyword "entel peru" --top-n 5
  python main.py facebook scrape --keyword "entel peru" --top-n 5

  # Con filtros de fecha
  python main.py tiktok scrape --username entel_peru --top-n 5 \
      --date-from 2025-01-01 --date-to 2025-12-31

  # Excluir comentarios/replies de usuarios especificos y limitar replies
  python main.py tiktok scrape --username entel_peru --top-n 5 \
      --exclude-users entel_peru otrousuario --max-replies 100

  # Solo clasificar comentarios ya guardados
  python main.py tiktok classify --input outputs/output_tiktok/comments.json

  # Buscar comentarios similares
  python main.py tiktok search --query "problema con la red" --top-k 10
"""
import argparse
import logging
import sys
from pathlib import Path

# Asegurar que src/ sea importable desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config, get_session_path, get_output_dir
from src.utils.io import save_json, timestamped_filename, setup_logging


def cmd_tiktok_auth(args, config):
    from src.scrapers.tiktok.auth import save_session
    session_file = str(get_session_path("tiktok", config))
    save_session(username=args.username, session_file=session_file, config=config)


def cmd_tiktok_scrape(args, config):
    from src.scrapers.tiktok.auth import check_session
    from src.scrapers.tiktok import scraper as tt
    from src.scrapers.tiktok import keyword_scraper as tt_kw
    session_file = str(get_session_path("tiktok", config))
    if not check_session(session_file):
        sys.exit(1)

    output_dir = get_output_dir("tiktok", config)

    exclude_users = args.exclude_users or []
    max_replies = args.max_replies

    if args.keyword:
        results = tt_kw.search_by_keyword(
            keyword=args.keyword,
            top_n=args.top_n,
            session_file=session_file,
            config=config,
            date_from=args.date_from,
            date_to=args.date_to,
            exclude_users=exclude_users,
            max_replies=max_replies,
        )
        base_name = args.keyword.replace(" ", "_")
    else:
        results = tt.search_by_user(
            username=args.username,
            top_n=args.top_n,
            session_file=session_file,
            config=config,
            date_from=args.date_from,
            date_to=args.date_to,
            exclude_users=exclude_users,
            max_replies=max_replies,
        )
        base_name = args.username

    # Separar videos y comentarios en archivos distintos (formato PRD)
    all_videos = [r["video"] for r in results]
    all_comments = []
    for r in results:
        for c in r["comments"]:
            c["video_id"] = r["video"].get("id", "")
            all_comments.append(c)

    ts_suffix = timestamped_filename(base_name)
    save_json(all_videos, output_dir, "videos.json")
    save_json(all_comments, output_dir, "comments.json")

    print(f"\nResultados guardados en: {output_dir}")
    print(f"  Videos       : {len(all_videos)}")
    print(f"  Comentarios  : {len(all_comments)}")

    if args.transcribe:
        _run_transcription(all_videos, config, output_dir)

    if args.classify:
        scrape_user = args.username if args.username else None
        _run_classification(all_comments, config, output_dir, platform="tiktok", exclude_users=[scrape_user] if scrape_user else [])


def cmd_facebook_auth(args, config):
    from src.scrapers.facebook.auth import save_session
    session_file = str(get_session_path("facebook", config))
    save_session(session_file=session_file, config=config)


def cmd_facebook_scrape(args, config):
    from src.scrapers.facebook.auth import check_session
    from src.scrapers.facebook import scraper as fb
    session_file = str(get_session_path("facebook", config))
    if not check_session(session_file):
        sys.exit(1)

    output_dir = get_output_dir("facebook", config)

    if args.keyword:
        results = fb.search_by_keyword(
            keyword=args.keyword,
            top_n=args.top_n,
            session_file=session_file,
            config=config,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        base_name = args.keyword.replace(" ", "_")
    else:
        results = fb.search_by_user(
            username=args.username,
            top_n=args.top_n,
            session_file=session_file,
            config=config,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        base_name = args.username

    all_posts = [r["post"] for r in results if r]
    all_comments = []
    for r in results:
        if not r:
            continue
        for c in r["comments"]:
            c["post_url"] = r["post"].get("url", "")
            all_comments.append(c)

    save_json(all_posts, output_dir, "videos.json")   # PRD llama 'videos' a posts tb
    save_json(all_comments, output_dir, "comments.json")

    print(f"\nResultados guardados en: {output_dir}")
    print(f"  Posts        : {len(all_posts)}")
    print(f"  Comentarios  : {len(all_comments)}")

    if args.classify:
        scrape_user = args.username if args.username else None
        _run_classification(all_comments, config, output_dir, platform="facebook", text_field="texto", exclude_users=[scrape_user] if scrape_user else [])


def cmd_classify(args, config):
    """Clasifica comentarios desde un archivo JSON ya guardado."""
    import json
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Archivo no encontrado: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        comments = json.load(f)

    output_dir = input_path.parent
    text_field = args.text_field or "comentario"
    exclude_users = args.exclude_users or [] if hasattr(args, 'exclude_users') else []
    _run_classification(comments, config, output_dir, platform=args.platform, text_field=text_field, exclude_users=exclude_users)


def cmd_clean(args, config):
    """
    Preprocesa los JSONs crudos de TikTok y Facebook para uno o varios operadores.

    TikTok:   outputs/{op}/mes_{periodo}/video_*.json       → preprocesado/VIDEO_CLEANER_*.json
    Facebook: outputs/{op}/mes_{periodo}/facebook/post_*.json → preprocesado/facebook_clean_*.json
    """
    from src.preprocessing.cleaner import clean_operator_videos, clean_operator_facebook

    operators = args.operators if args.operators else ["claro", "entel", "movistar", "bitel"]
    base_dir = Path(args.base_dir)

    # Detectar periodos disponibles si no se indicó uno
    for op in operators:
        op_dir = base_dir / op
        if not op_dir.exists():
            print(f"  [{op}] carpeta no encontrada, saltando")
            continue

        periodos = sorted(
            d.name.replace("mes_", "")
            for d in op_dir.iterdir()
            if d.is_dir() and d.name.startswith("mes_")
        )
        if args.periodo:
            periodos = [args.periodo]

        if not periodos:
            print(f"  [{op}] sin carpetas mes_YYYYMM, saltando")
            continue

        print(f"\n── {op.upper()} ──")
        for periodo in periodos:
            print(f"  Periodo: {periodo}")
            clean_operator_videos(op, periodo, base_dir=base_dir)
            clean_operator_facebook(op, periodo, base_dir=base_dir)


def cmd_classify_files(args, config):
    """
    Clasifica los archivos preprocesados (VIDEO_CLEANER_* y facebook_clean_*) de uno o
    varios operadores y guarda los resultados en clasificado/.
    """
    from src.pipeline import run_classify_from_files
    from src.word_clouds.phrase_cloud import generate_phrase_jsons_for_operator

    operators = args.operators if args.operators else ["claro", "entel", "movistar", "bitel"]
    base_dir = Path(args.base_dir)

    for op in operators:
        print(f"\n── {op.upper()} ──")
        resumen = run_classify_from_files(op, config=config, base_dir=base_dir)
        if resumen["archivos"] == 0:
            print(f"  Sin archivos preprocesados en outputs/{op}/preprocesado/")
            continue
        print(f"  Clasificados : {resumen['comentarios']} comentarios en {resumen['archivos']} archivos")
        counts = {k: v for k, v in resumen.items() if k not in ("operador", "archivos", "comentarios")}
        for label, n in counts.items():
            print(f"    {label:15s}: {n}")

        paths = generate_phrase_jsons_for_operator(op)
        print(f"  Phrase clouds: {len(paths)} archivos → outputs_front/{op}/")


def cmd_process(args, config):
    """
    Clasifica los VIDEO_CLEANER de un operador y genera los phrase clouds para el dashboard.
    Equivale a: run_classify_from_files + generate_phrase_jsons_for_operator
    """
    from src.pipeline import run_classify_from_files
    from src.word_clouds.phrase_cloud import generate_phrase_jsons_for_operator

    operators = args.operators if args.operators else ["claro", "entel", "movistar", "bitel"]

    for op in operators:
        print(f"\n── {op.upper()} ──")
        resumen = run_classify_from_files(op)
        if resumen["archivos"] == 0:
            print(f"  Sin archivos preprocesados en outputs/{op}/preprocesado/")
            continue
        print(f"  Clasificados : {resumen['comentarios']} comentarios en {resumen['archivos']} videos")

        paths = generate_phrase_jsons_for_operator(op)
        print(f"  Phrase clouds: {len(paths)} archivos → outputs_front/{op}/")


def cmd_process_all(args, config):
    """Clasifica todos los operadores y genera phrase clouds para el dashboard."""
    from src.pipeline import run_process_all
    resumenes = run_process_all()
    for r in resumenes:
        print(f"  {r['operador']}: {r['comentarios']} comentarios en {r['archivos']} videos")
    print("\nListo. Corre el dashboard con: .venv/bin/streamlit run dashboard.py")


def cmd_search(args, config):
    """Busca comentarios similares a una query en el vector store."""
    from src.retrieval.classifier import CommentClassifier
    clf = CommentClassifier(config)
    results = clf.search_similar(query=args.query, platform=args.platform, top_k=args.top_k)

    print(f"\nResultados para: '{args.query}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['clasificacion'].upper()}] {r['texto'][:100]}")
        print(f"   Distancia: {r['distancia']:.4f}\n")


def _run_transcription(videos: list, config: dict, output_dir: Path):
    from src.transcription.transcriber import batch_transcribe
    print("\nTranscribiendo videos...")
    transcripts = batch_transcribe(videos, config, output_dir)
    save_json(transcripts, output_dir, "transcripts.json")
    print(f"  Transcripciones guardadas: {len(transcripts)}")


def _run_classification(
    comments: list, config: dict, output_dir: Path,
    platform: str = "tiktok", text_field: str = "comentario", exclude_users: list[str] | None = None
):
    from src.preprocessing.cleaner import clean_comments
    from src.retrieval.classifier import CommentClassifier

    print("\nLimpiando y deduplicando comentarios...")
    comments_clean = clean_comments(comments, config, text_field=text_field, exclude_users=exclude_users)

    print("\nClasificando comentarios...")
    clf = CommentClassifier(config)
    classified = clf.classify_comments(comments_clean, text_field=text_field, platform=platform)
    save_json(classified, output_dir, "comments.json")  # Sobreescribe con clasificacion incluida

    emb_dir = output_dir / "embeddings"
    clf.save_embeddings(classified, emb_dir, text_field=text_field)
    print(f"  Comentarios clasificados: {len(classified)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Pipeline de scraping y clasificacion para TikTok y Facebook",
    )
    parser.add_argument("--config", default=None, help="Ruta a config.yaml personalizado")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="platform", required=True, metavar="{tiktok,facebook,process}")

    # ---- TikTok ----
    tt_parser = subparsers.add_parser("tiktok", help="Comandos para TikTok")
    tt_sub = tt_parser.add_subparsers(dest="command", required=True)

    tt_auth = tt_sub.add_parser("auth", help="Guardar sesion de TikTok")
    tt_auth.add_argument("--username", required=True, help="Usuario de TikTok (sin @)")

    tt_scrape = tt_sub.add_parser("scrape", help="Extraer videos y comentarios")
    tt_mode = tt_scrape.add_mutually_exclusive_group(required=True)
    tt_mode.add_argument("--username", help="Usuario de TikTok (sin @)")
    tt_mode.add_argument("--keyword", help="Keyword de busqueda")
    tt_scrape.add_argument("--top-n", type=int, default=5, help="Cantidad de videos a extraer")
    tt_scrape.add_argument("--date-from", default=None, help="Fecha inicio YYYY-MM-DD")
    tt_scrape.add_argument("--date-to", default=None, help="Fecha fin YYYY-MM-DD")
    tt_scrape.add_argument(
        "--exclude-users", nargs="+", metavar="USER", default=None,
        help="Uno o mas unique_id de TikTok cuyos comentarios/replies se excluyen",
    )
    tt_scrape.add_argument(
        "--max-replies", type=int, default=None, metavar="N",
        help="Maximo de replies a conservar por comentario (sin limite por defecto)",
    )
    tt_scrape.add_argument("--transcribe", action="store_true", help="Transcribir videos")
    tt_scrape.add_argument("--classify", action="store_true", help="Clasificar comentarios")

    tt_classify = tt_sub.add_parser("classify", help="Clasificar comentarios desde archivo")
    tt_classify.add_argument("--input", required=True, help="Ruta al JSON de comentarios")
    tt_classify.add_argument("--text-field", default="comentario")
    tt_classify.add_argument("--platform", default="tiktok")
    tt_classify.add_argument(
        "--exclude-users", nargs="+", metavar="USER", default=None,
        help="Uno o mas usuario_id a excluir de la clasificacion",
    )

    tt_search = tt_sub.add_parser("search", help="Buscar comentarios similares")
    tt_search.add_argument("--query", required=True, help="Texto de busqueda")
    tt_search.add_argument("--top-k", type=int, default=10)
    tt_search.add_argument("--platform", default="tiktok")

    # ---- Facebook ----
    fb_parser = subparsers.add_parser("facebook", help="Comandos para Facebook")
    fb_sub = fb_parser.add_subparsers(dest="command", required=True)

    fb_auth = fb_sub.add_parser("auth", help="Guardar sesion de Facebook")

    fb_scrape = fb_sub.add_parser("scrape", help="Extraer posts y comentarios")
    fb_mode = fb_scrape.add_mutually_exclusive_group(required=True)
    fb_mode.add_argument("--username", help="Nombre de pagina de Facebook")
    fb_mode.add_argument("--keyword", help="Keyword de busqueda")
    fb_scrape.add_argument("--top-n", type=int, default=None, help="Cantidad de posts a extraer (sin limite por defecto)")
    fb_scrape.add_argument("--date-from", default=None, help="Fecha inicio YYYY-MM-DD")
    fb_scrape.add_argument("--date-to", default=None, help="Fecha fin YYYY-MM-DD")
    fb_scrape.add_argument("--classify", action="store_true", help="Clasificar comentarios")

    fb_classify = fb_sub.add_parser("classify", help="Clasificar comentarios desde archivo")
    fb_classify.add_argument("--input", required=True, help="Ruta al JSON de comentarios")
    fb_classify.add_argument("--text-field", default="texto")
    fb_classify.add_argument("--platform", default="facebook")
    fb_classify.add_argument(
        "--exclude-users", nargs="+", metavar="USER", default=None,
        help="Uno o mas usuario_id a excluir de la clasificacion",
    )

    fb_search = fb_sub.add_parser("search", help="Buscar comentarios similares")
    fb_search.add_argument("--query", required=True)
    fb_search.add_argument("--top-k", type=int, default=10)
    fb_search.add_argument("--platform", default="facebook")

    # ---- clean (preprocesamiento TikTok + Facebook) ----
    clean_parser = subparsers.add_parser(
        "clean",
        help="Preprocesa JSONs crudos de TikTok y Facebook → preprocesado/",
    )
    clean_parser.add_argument(
        "--operators", nargs="+", metavar="OP",
        default=None,
        help="Operadores a procesar (default: claro entel movistar bitel)",
    )
    clean_parser.add_argument(
        "--periodo", default=None, metavar="YYYYMM",
        help="Periodo específico (default: todos los mes_* encontrados)",
    )
    clean_parser.add_argument(
        "--base-dir", default="outputs", metavar="DIR",
        help="Directorio base de outputs (default: outputs)",
    )

    # ---- classify-files (clasificación desde preprocesado/) ----
    clf_parser = subparsers.add_parser(
        "classify-files",
        help="Clasifica archivos preprocesados (VIDEO_CLEANER_* y facebook_clean_*) → clasificado/",
    )
    clf_parser.add_argument(
        "--operators", nargs="+", metavar="OP",
        default=None,
        help="Operadores a procesar (default: claro entel movistar bitel)",
    )
    clf_parser.add_argument(
        "--base-dir", default="outputs", metavar="DIR",
        help="Directorio base de outputs (default: outputs)",
    )

    # ---- Process (classify + phrase clouds para todos los operadores) ----
    subparsers.add_parser(
        "process",
        help="Clasifica todos los operadores y genera phrase clouds para el dashboard",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    config = load_config(args.config)

    # comandos sin subcomando, se despachan directo
    if args.platform == "process":
        cmd_process_all(args, config)
        return
    if args.platform == "clean":
        cmd_clean(args, config)
        return
    if args.platform == "classify-files":
        cmd_classify_files(args, config)
        return

    dispatch = {
        ("tiktok", "auth"):     cmd_tiktok_auth,
        ("tiktok", "scrape"):   cmd_tiktok_scrape,
        ("tiktok", "classify"): cmd_classify,
        ("tiktok", "search"):   cmd_search,
        ("facebook", "auth"):     cmd_facebook_auth,
        ("facebook", "scrape"):   cmd_facebook_scrape,
        ("facebook", "classify"): cmd_classify,
        ("facebook", "search"):   cmd_search,
    }

    handler = dispatch.get((args.platform, args.command))
    if not handler:
        parser.print_help()
        sys.exit(1)

    handler(args, config)


if __name__ == "__main__":
    main()
