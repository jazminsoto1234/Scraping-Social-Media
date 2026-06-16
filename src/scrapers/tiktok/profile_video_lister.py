"""
Paso 1 del pipeline: dado un rango de fechas, navega el perfil de cada operador
y guarda la lista de videos del período en outputs/listas/{YYYYMM}/{operador}_videos.json.

Uso:
    python src/scrapers/tiktok/profile_video_lister.py --date-from 2025-05-01 --date-to 2025-05-31
"""
from playwright.sync_api import sync_playwright
from pathlib import Path
import argparse
import json
import random
import logging
import yaml
from datetime import datetime

logging.basicConfig(
    filename="profile_video_lister.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)


def _cargar_config() -> dict:
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_profile_videos(page, profile_url: str, username: str, date_from: str, date_to: str) -> list:
    """
    Navega al perfil y retorna la lista de videos publicados entre date_from y date_to.
    Retorna: [{"id": "...", "url": "...", "fecha": "YYYY-MM-DD", "comment_count": N}, ...]
    """

    videos_encontrados = []
    seen_ids = set()
    fuera_rango_count = 0
    parar = False

    def handle_response(response):
        nonlocal fuera_rango_count, parar
        if "api/post/item_list" not in response.url:
            return
        try:
            data = response.json()
        except Exception:
            return

        items = data.get("itemList") or data.get("items") or []
        for item in items:
            video_id = str(item.get("id", ""))
            if not video_id or video_id in seen_ids:
                continue

            create_time = item.get("createTime", 0)
            if not create_time:
                continue

            fecha = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d")

            stats = item.get("stats") or {}
            try:
                comment_count = int(stats.get("commentCount", 0) or 0)
            except (TypeError, ValueError):
                comment_count = 0

            if fecha > date_to:
                logging.debug(f"Video {video_id} fecha {fecha} > {date_to}, ignorado")
                fuera_rango_count = 0
                continue

            if date_from <= fecha <= date_to:
                seen_ids.add(video_id)
                url = f"https://www.tiktok.com/@{username}/video/{video_id}"
                videos_encontrados.append({
                    "id": video_id,
                    "url": url,
                    "fecha": fecha,
                    "comment_count": comment_count,
                })
                fuera_rango_count = 0
                logging.info(f"Video encontrado: {video_id} fecha={fecha} comments={comment_count}")
            else:
                fuera_rango_count += 1
                logging.debug(f"Video {video_id} fecha {fecha} < {date_from}, fuera_rango_count={fuera_rango_count}")
                if fuera_rango_count >= 10:
                    parar = True

    # Registrar handler ANTES de navegar para capturar el primer batch SSR
    page.on("response", handle_response)

    print(f"  Navegando a {profile_url} ...")
    # networkidle espera a que los requests iniciales (incluido item_list SSR) se completen
    page.goto(profile_url, wait_until="networkidle", timeout=40000)
    page.wait_for_timeout(3000)

    if "login" in page.url or "passport" in page.url:
        print("  TikTok pide login — inicia sesión en el navegador (tienes 60s)...")
        for _ in range(60):
            page.wait_for_timeout(1000)
            if "login" not in page.url and "passport" not in page.url:
                print("  Login detectado, continuando...")
                page.goto(profile_url, wait_until="networkidle", timeout=40000)
                page.wait_for_timeout(3000)
                break
        else:
            print("  Timeout esperando login — abortando")
            return []

    # Pausa extra para que el primer batch SSR dispare el handler antes de scrollear
    page.wait_for_timeout(2000)
    print(f"  Videos capturados en carga inicial: {len(videos_encontrados)}")

    sin_cambios = 0
    prev_count = len(videos_encontrados)

    while True:
        if parar:
            print(f"  Parada temprana: 10 videos consecutivos fuera del rango")
            break

        page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
        # Espera más larga para dar tiempo a la API de responder antes de contar sin_cambios
        page.wait_for_timeout(random.randint(2000, 3000))

        if len(videos_encontrados) > prev_count:
            sin_cambios = 0
            prev_count = len(videos_encontrados)
            print(f"  Videos en rango: {len(videos_encontrados)}")
        else:
            sin_cambios += 1
            # Umbral más alto (12) para tolerar latencia de red antes de considerar fin
            if sin_cambios >= 12:
                print(f"  Sin nuevos videos en 12 scrolls consecutivos — terminando")
                break

    videos_encontrados.sort(key=lambda v: v["fecha"], reverse=True)
    return videos_encontrados


def _validar_fecha(valor: str, nombre: str) -> str:
    """Valida formato YYYY-MM-DD y normaliza (agrega ceros si faltan)."""
    try:
        dt = datetime.strptime(valor, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"--{nombre} debe tener formato YYYY-MM-DD, recibido: '{valor}'")


def main():
    parser = argparse.ArgumentParser(description="Paso 1: listar videos de perfil por rango de fechas")
    parser.add_argument("--date-from", required=True, metavar="YYYY-MM-DD", help="Fecha inicio (inclusive)")
    parser.add_argument("--date-to",   required=True, metavar="YYYY-MM-DD", help="Fecha fin (inclusive)")
    args = parser.parse_args()

    try:
        date_from = _validar_fecha(args.date_from, "date-from")
        date_to   = _validar_fecha(args.date_to,   "date-to")
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    config = _cargar_config()
    operators = config.get("operators", {})
    if not operators:
        print("ERROR: No se encontró la sección 'operators' en config.yaml")
        return

    periodo = date_from[:7].replace("-", "")
    listas_dir = Path("outputs") / "listas" / periodo
    listas_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        print("Conectando al Chrome real via CDP (localhost:9222)...")
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"No se pudo conectar: {e}")
            print("  Asegurate de tener Chrome abierto con --remote-debugging-port=9222")
            return

        try:
            context = browser.contexts[0]

            for nombre, username in operators.items():
                profile_url = f"https://www.tiktok.com/@{username}"
                print(f"\nOperador: {nombre} (@{username})")

                page = context.new_page()
                try:
                    videos = list_profile_videos(page, profile_url, username, date_from, date_to)
                finally:
                    page.close()

                output = {
                    "operador":     nombre,
                    "perfil":       username,
                    "date_from":    date_from,
                    "date_to":      date_to,
                    "total_videos": len(videos),
                    "videos":       videos,
                }

                out_path = listas_dir / f"{nombre}_videos.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"  Guardado: {out_path}  ({len(videos)} videos)")

                print(f"\nPerfil: {username}  |  {date_from} -> {date_to}  |  {len(videos)} videos encontrados")
                if videos:
                    print(f"\n  {'#':<4} {'Fecha':<12} {'Comentarios':>12}   URL")
                    for i, v in enumerate(videos, 1):
                        url_corta = v["url"][:80]
                        print(f"  {i:<4} {v['fecha']:<12} {v['comment_count']:>12}   {url_corta}")

        finally:
            browser.close()


if __name__ == "__main__":
    main()
