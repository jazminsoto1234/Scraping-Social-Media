"""
Paso 2 del pipeline: lee los JSON producidos por profile_video_lister.py y scrapea
los comentarios de cada video, guardando en outputs/{operador}/mes_{YYYYMM}/video_{ID}.json.

Requisito previo: correr profile_video_lister.py para generar outputs/listas/{YYYYMM}/*.json

Uso:
    python src/scrapers/tiktok/operadores_scraper.py --periodo 202505
    python src/scrapers/tiktok/operadores_scraper.py --periodo 202505 --operators entel claro
"""
from playwright.sync_api import sync_playwright
from pathlib import Path
import argparse
import json
import re
import time
import random
import logging
import yaml
from datetime import datetime

logging.basicConfig(
    filename="operadores_scraper.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)


# ---------------------------------------------------------------------------
# Tarea 1: cargar config
# ---------------------------------------------------------------------------

def _cargar_config() -> dict:
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tarea 2: _guardar_video — copia parametrizada de guardar_incremental
# ---------------------------------------------------------------------------

def _guardar_video(all_comments, all_replies, video_url: str, output_path: Path, operador: str, fecha_video: str):
    m = re.match(r"https?://(?:www\.)?tiktok\.com/@([^/]+)/video/(\d+)", video_url)
    video_id = m.group(2) if m else "unknown"
    autor = m.group(1) if m else "unknown"

    video_meta = {
        "id":    video_id,
        "autor": autor,
        "url":   video_url,
    }

    comentarios_simplificados = []
    for comment in all_comments:
        user = comment.get("user") or {}
        comment_id = str(comment.get("cid", ""))
        respuestas_simples = []

        for reply in (comment.get("reply_comment") or []):
            ru = reply.get("user") or {}
            ts = reply.get("create_time", 0)
            respuestas_simples.append({
                "id":         str(reply.get("cid", "")),
                "usuario_id": ru.get("unique_id", "unknown"),
                "nickname":   ru.get("nickname", ""),
                "comentario": reply.get("text", ""),
                "likes":      reply.get("digg_count", 0),
                "fecha":      datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            })

        for reply in (all_replies.get(comment_id) or []):
            ru = reply.get("user") or {}
            ts = reply.get("create_time", 0)
            respuestas_simples.append({
                "id":         str(reply.get("cid", "")),
                "usuario_id": ru.get("unique_id", "unknown"),
                "nickname":   ru.get("nickname", ""),
                "comentario": reply.get("text", ""),
                "likes":      reply.get("digg_count", 0),
                "fecha":      datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            })

        ts = comment.get("create_time", 0)
        comentarios_simplificados.append({
            "id":            comment_id,
            "usuario_id":    user.get("unique_id", "unknown"),
            "nickname":      user.get("nickname", ""),
            "comentario":    comment.get("text", ""),
            "likes":         comment.get("digg_count", 0),
            "fecha":         datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            "num_respuestas": comment.get("reply_comment_total", 0),
            "respuestas":    respuestas_simples,
        })

    output = {
        "video":             video_meta,
        "operador":          operador,
        "fecha_video":       fecha_video,
        "total_comentarios": len(comentarios_simplificados),
        "total_respuestas":  sum(len(c["respuestas"]) for c in comentarios_simplificados),
        "total_items":       len(comentarios_simplificados) + sum(len(c["respuestas"]) for c in comentarios_simplificados),
        "comentarios":       comentarios_simplificados,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.debug(f"Guardado: {output['total_items']} items -> {output_path}")


# ---------------------------------------------------------------------------
# Tarea 3: _scrape_video_comments — copia parametrizada de scraper_main
# ---------------------------------------------------------------------------

def _scrape_video_comments(page, video_url: str, output_path: Path, operador: str, fecha_video: str):
    m = re.match(r"https?://(?:www\.)?tiktok\.com/@([^/]+)/video/(\d+)", video_url)
    video_id = m.group(2) if m else "unknown"

    all_comments = []
    all_replies = {}
    seen_comment_ids = set()
    seen_reply_ids = set()

    def handle_response(response):
        url = response.url
        guardado = False

        if "api/comment/list" in url and "reply" not in url:
            try:
                data = response.json()
                batch = data.get("comments") or []
                nuevos = 0
                for c in batch:
                    cid = str(c.get("cid", ""))
                    if cid and cid not in seen_comment_ids:
                        seen_comment_ids.add(cid)
                        all_comments.append(c)
                        nuevos += 1
                if nuevos:
                    logging.info(f"+{nuevos} comentarios, total: {len(all_comments)}")
                    guardado = True
            except Exception as e:
                logging.error(f"Error parseando comentarios: {e}")

        if "/api/comment/list/reply/" in url:
            try:
                data = response.json()
                replies = data.get("comments") or []
                if replies:
                    match = re.search(r"comment_id=(\d+)", url)
                    if match:
                        parent_id = match.group(1)
                        if parent_id not in all_replies:
                            all_replies[parent_id] = []
                        nuevos_r = 0
                        for r in replies:
                            rid = str(r.get("cid", ""))
                            if rid and rid not in seen_reply_ids:
                                seen_reply_ids.add(rid)
                                all_replies[parent_id].append(r)
                                nuevos_r += 1
                        if nuevos_r:
                            logging.info(f"+{nuevos_r} replies para {parent_id}")
                            guardado = True
            except Exception as e:
                logging.error(f"Error parseando replies: {e}")

        if guardado:
            try:
                _guardar_video(all_comments, all_replies, video_url, output_path, operador, fecha_video)
            except Exception as e:
                logging.error(f"Error guardado incremental: {e}")

    page.on("response", handle_response)

    # Navegar al video o reusar si ya está abierto
    if video_id not in page.url:
        print(f"    Navegando al video {video_id}...")
        page.goto(video_url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(4000)

    if "login" in page.url or "passport" in page.url:
        print("    TikTok pide login — inicia sesión en el navegador (tienes 60s)...")
        for _ in range(60):
            page.wait_for_timeout(1000)
            if "login" not in page.url and "passport" not in page.url:
                print("    Login detectado, continuando...")
                page.goto(video_url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(4000)
                break
        else:
            print("    Timeout esperando login — abortando video")
            return

    page.wait_for_timeout(3000)

    resultado_tab = page.evaluate("""
        () => {
            const candidates = [
                ...document.querySelectorAll('[data-e2e="comment-tab"]'),
                ...document.querySelectorAll('[data-e2e="comments-tab"]'),
                ...document.querySelectorAll('[data-e2e="browse-comment-tab"]'),
            ];
            for (const el of candidates) {
                el.click();
                return 'data-e2e:' + el.getAttribute('data-e2e');
            }
            for (const el of document.querySelectorAll('[role="tab"]')) {
                const txt = el.textContent.trim();
                if ((txt === 'Comentarios' || txt === 'Comments') && el.offsetParent !== null) {
                    el.click();
                    return 'role-tab-visible:' + txt;
                }
            }
            for (const sel of [
                '[data-e2e="comment-icon"]',
                '[data-e2e="browse-comment"]',
                'a[href*="comment"]',
            ]) {
                const el = document.querySelector(sel);
                if (el && el.offsetParent !== null) {
                    el.click();
                    return 'icon:' + sel;
                }
            }
            return null;
        }
    """)
    logging.debug(f"Tab click: {resultado_tab}")

    panel_listo = False
    for _w in range(16):
        page.wait_for_timeout(500)
        _test = page.evaluate("""
            () => {
                const divs = document.querySelectorAll('div');
                for (const d of divs) {
                    if (d.scrollHeight > 500 && d.clientHeight > 100
                            && d.clientHeight < d.scrollHeight) {
                        const r = d.getBoundingClientRect();
                        if (r.y > 150 && r.height > 100)
                            return true;
                    }
                }
                return false;
            }
        """)
        if _test:
            panel_listo = True
            logging.debug(f"Panel detectado tras {(_w + 1) * 0.5:.1f}s")
            break
    if not panel_listo:
        logging.warning("Panel scrollable no detectado, continuando igual")

    def centro_panel():
        return page.evaluate("""
            () => {
                const divs = document.querySelectorAll('div');
                let best = null, bestH = 0;
                for (const d of divs) {
                    if (d.scrollHeight > 500 && d.clientHeight > 100
                        && d.clientHeight < d.scrollHeight
                        && d.scrollHeight > bestH) {
                        const r = d.getBoundingClientRect();
                        if (r.y > 150 && r.height > 100) {
                            best = d; bestH = d.scrollHeight;
                        }
                    }
                }
                if (!best) return null;
                const r = best.getBoundingClientRect();
                return {
                    x: r.x + r.width / 2,
                    y: r.y + r.height / 2,
                    scrollTop: best.scrollTop,
                };
            }
        """)

    def _get_scroll_top():
        return page.evaluate("""
            () => {
                const divs = document.querySelectorAll('div');
                let best = null, bestH = 0;
                for (const d of divs) {
                    if (d.scrollHeight > 500 && d.clientHeight > 100
                        && d.clientHeight < d.scrollHeight
                        && d.scrollHeight > bestH) {
                        const r = d.getBoundingClientRect();
                        if (r.y > 150 && r.height > 100) {
                            best = d; bestH = d.scrollHeight;
                        }
                    }
                }
                return best ? best.scrollTop : null;
            }
        """)

    def scroll_panel(delta, px, py):
        antes = _get_scroll_top()
        if antes is None:
            return None, None
        page.mouse.move(px, py)
        page.mouse.wheel(0, delta)
        page.wait_for_timeout(60)
        despues = _get_scroll_top()
        for _retry in range(2):
            if despues is not None and despues != antes:
                break
            logging.warning(f"  Wheel no movió (intento {_retry + 1}), reposicionando...")
            page.mouse.move(px, py)
            page.wait_for_timeout(80)
            page.mouse.wheel(0, delta)
            page.wait_for_timeout(60)
            despues = _get_scroll_top()
        if despues is not None and despues == antes:
            logging.warning("  Wheel sin efecto tras 2 reintentos")
        return antes, despues

    _c = centro_panel()
    if _c:
        _px, _py = _c["x"], _c["y"]
    else:
        _px, _py = 0.0, 0.0
    logging.debug(f"Panel calibrado: x={_px:.0f} y={_py:.0f}")

    # ── Pasada 1+2: scroll continuo para cargar comentarios padre ──
    print("    Cargando comentarios padre...")
    sin_cambios = 0
    prev_count = 0
    grupo_num = 0

    while True:
        grupo_num += 1
        pasos = random.randint(5, 10)
        page.mouse.move(_px, _py)
        for _ in range(pasos):
            delta = random.randint(150, 280)
            scroll_panel(delta, _px, _py)
            page.wait_for_timeout(random.randint(200, 450))

        page.wait_for_timeout(random.randint(400, 800))

        if grupo_num % random.randint(5, 8) == 0:
            pausa_larga = random.randint(1500, 3000)
            logging.debug(f"  Pausa larga ({pausa_larga}ms) en grupo {grupo_num}")
            page.wait_for_timeout(pausa_larga)

        if len(all_comments) > prev_count:
            sin_cambios = 0
            print(f"    comentarios: {len(all_comments)}")
            _guardar_video(all_comments, all_replies, video_url, output_path, operador, fecha_video)
            prev_count = len(all_comments)
        else:
            sin_cambios += 1

            if sin_cambios == 5:
                print("    Sin cambios x5 — recalibrando panel...")
                _c2 = centro_panel()
                if _c2:
                    _px, _py = _c2["x"], _c2["y"]
                page.mouse.move(_px, _py)
                for _ in range(3):
                    page.mouse.move(_px, _py)
                    for _ in range(random.randint(5, 8)):
                        scroll_panel(random.randint(150, 280), _px, _py)
                        page.wait_for_timeout(random.randint(200, 450))
                    page.wait_for_timeout(random.randint(400, 800))
                if len(all_comments) > prev_count:
                    sin_cambios = 0
                    prev_count = len(all_comments)
                    print(f"    Recovery exitoso: {len(all_comments)} comentarios")

            if sin_cambios >= 10:
                print(f"    Scroll terminado ({len(all_comments)} comentarios)")
                break

    # ── Pasada 3: expandir replies ──
    print("    Expandiendo replies...")
    prev_replies = 0

    # Selectores específicos de replies — sin genéricos tipo "Ver más" que
    # pueden matchear links patrocinados fuera del panel de comentarios.
    reply_selectors = [
        "text=/ver \\d+ respuesta/i",
        "text=/ver \\d+ repl/i",
        "text=/\\d+ respuesta/i",
        "text=/\\d+ repl/i",
    ]

    def expandir_replies_visibles():
        clicked_local = 0
        for _ in range(6):
            nuevos_clicks = 0
            for selector in reply_selectors:
                try:
                    for elem in page.locator(selector).all():
                        try:
                            if not elem.is_visible(timeout=300):
                                continue
                            # Verificar que el elemento esté dentro del panel de
                            # comentarios (bounding box dentro del área visible)
                            # antes de clickear, para evitar links patrocinados.
                            box = elem.bounding_box()
                            if box is None or box["y"] < 100:
                                continue
                            page.wait_for_timeout(random.randint(150, 350))
                            elem.click()
                            nuevos_clicks += 1
                            clicked_local += 1
                            page.wait_for_timeout(random.randint(700, 1300))
                        except Exception:
                            pass
                except Exception:
                    pass
            if nuevos_clicks == 0:
                break
            page.wait_for_timeout(random.randint(500, 1000))
        return clicked_local

    for _ in range(60):
        scroll_panel(-500, _px, _py)
        page.wait_for_timeout(20)
    page.evaluate("""
        () => {
            const divs = document.querySelectorAll('div');
            let best = null, bestH = 0;
            for (const d of divs) {
                if (d.scrollHeight > 500 && d.clientHeight > 100
                    && d.clientHeight < d.scrollHeight
                    && d.scrollHeight > bestH) {
                    const r = d.getBoundingClientRect();
                    if (r.y > 150 && r.height > 100) {
                        best = d; bestH = d.scrollHeight;
                    }
                }
            }
            if (best) best.scrollTop = 0;
        }
    """)
    page.wait_for_timeout(800)

    TIMEOUT_INACTIVIDAD_REPLIES = 3 * 60
    ultimo_progreso_replies = time.time()

    while True:
        if (time.time() - ultimo_progreso_replies) > TIMEOUT_INACTIVIDAD_REPLIES:
            print(f"    Replies: JSON sin crecer 3 min — saliendo ({sum(len(r) for r in all_replies.values())} replies)")
            break

        page.mouse.move(_px, _py)
        for _ in range(random.randint(4, 7)):
            delta = random.randint(120, 220)
            scroll_panel(delta, _px, _py)
            page.wait_for_timeout(random.randint(200, 450))

        expandir_replies_visibles()

        page.wait_for_timeout(random.randint(600, 1200))

        current_replies = sum(len(r) for r in all_replies.values())
        if current_replies > prev_replies:
            ultimo_progreso_replies = time.time()
            print(f"    replies: {current_replies}")
            _guardar_video(all_comments, all_replies, video_url, output_path, operador, fecha_video)
        prev_replies = current_replies

    total_replies = sum(len(r) for r in all_replies.values())
    print(f"    Terminado: {len(all_comments)} comentarios, {total_replies} replies")

    _guardar_video(all_comments, all_replies, video_url, output_path, operador, fecha_video)

    # Remover el handler para no acumular en la misma page entre videos
    page.remove_listener("response", handle_response)


# ---------------------------------------------------------------------------
# Tarea 4: _cargar_lista_videos — lee el JSON del Paso 1
# ---------------------------------------------------------------------------

def _cargar_lista_videos(operator_name: str, periodo: str) -> dict:
    lista_path = Path("outputs") / "listas" / periodo / f"{operator_name}_videos.json"
    if not lista_path.exists():
        raise FileNotFoundError(
            f"Falta {lista_path}. Corre primero profile_video_lister.py"
        )
    with open(lista_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tarea 5: _scrape_operator — orquesta un operador
# ---------------------------------------------------------------------------

def _scrape_operator(operator_name: str, periodo: str, context) -> dict:
    data = _cargar_lista_videos(operator_name, periodo)
    videos = data.get("videos", [])
    mes_label = f"mes_{periodo}"

    # Pre-scan: contar cuántos ya alcanzan el 80% de comment_count del lister
    ya_completos = 0
    for video in videos:
        comment_count = int(video.get("comment_count", 0) or 0)
        output_path = Path("outputs") / operator_name / mes_label / f"video_{video['id']}.json"
        if output_path.exists() and comment_count > 0:
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existente = json.load(f)
                if existente.get("total_items", 0) >= comment_count * 0.8:
                    ya_completos += 1
            except Exception:
                pass

    pendientes = len(videos) - ya_completos
    print(f"\n{'='*60}")
    print(f"Operador: {operator_name}  |  periodo: {periodo}")
    print(f"  Total videos : {len(videos)}")
    print(f"  Ya completos : {ya_completos}  (se saltaran)")
    print(f"  Pendientes   : {pendientes}  (se scrapearan)")
    print(f"{'='*60}")

    scrapeados = 0
    saltados = 0

    for i, video in enumerate(videos, 1):
        comment_count = int(video.get("comment_count", 0) or 0)
        output_path = Path("outputs") / operator_name / mes_label / f"video_{video['id']}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i}/{len(videos)}] {video['url']}")
        print(f"  Fecha: {video.get('fecha', '')}  |  Comentarios esperados: {comment_count}")

        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existente = json.load(f)
                total_items = existente.get("total_items", 0)
                if comment_count > 0:
                    umbral = comment_count * 0.8
                    if total_items >= umbral:
                        print(f"  SALTADO (total_items {total_items} >= {umbral:.0f})")
                        saltados += 1
                        continue
                    else:
                        print(f"  Incompleto (total_items {total_items} < {umbral:.0f}) — re-scrapeando")
                else:
                    print("  comment_count desconocido (0) — re-scrapeando para asegurar")
            except Exception:
                pass

        page = context.new_page()
        try:
            _scrape_video_comments(page, video["url"], output_path, operator_name, video["fecha"])
            scrapeados += 1
        finally:
            page.close()

        if i < len(videos):
            pausa = random.randint(3, 6)
            print(f"  Pausa {pausa}s antes del siguiente video...")
            time.sleep(pausa)

    return {
        "operador":    operator_name,
        "encontrados": len(videos),
        "scrapeados":  scrapeados,
        "saltados":    saltados,
    }


# ---------------------------------------------------------------------------
# Tarea 6: scrape_all_operators — entry point
# ---------------------------------------------------------------------------

def scrape_all_operators(periodo: str, operators: list = None):
    config = _cargar_config()
    todos = config.get("operators", {})

    if operators is not None:
        todos = {k: v for k, v in todos.items() if k in operators}

    if not todos:
        print("ERROR: No hay operadores configurados (revisa config.yaml o el filtro pasado)")
        return

    resultados = []

    with sync_playwright() as p:
        print("Conectando al Chrome real via CDP (localhost:9222)...")
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"No se pudo conectar: {e}")
            print("  Asegurate de tener Chrome abierto con --remote-debugging-port=9222")
            return

        # NO cerrar el browser: es el Chrome real del usuario conectado vía CDP.
        # Cada operador ya cierra su propia pestaña en _scrape_operator (page.close()).
        # Al salir del `with sync_playwright()` Playwright se desconecta sin matar Chrome.
        context = browser.contexts[0]
        for operator_name in todos:
            resultado = _scrape_operator(operator_name, periodo, context)
            resultados.append(resultado)

    print(f"\n{'='*60}")
    print(f"RESUMEN — periodo {periodo}")
    print(f"{'='*60}")
    print(f"  {'Operador':<12} {'Encontrados':>12} {'Scrapeados':>12} {'Saltados':>10}")
    print(f"  {'-'*48}")
    for r in resultados:
        print(f"  {r['operador']:<12} {r['encontrados']:>12} {r['scrapeados']:>12} {r['saltados']:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paso 2: scrapear comentarios de videos listados")
    parser.add_argument("--periodo",   required=True, metavar="YYYYMM",  help="Período a procesar, ej: 202505")
    parser.add_argument("--operators", nargs="+",     metavar="NOMBRE",  help="Filtrar operadores (opcional), ej: entel claro")
    args = parser.parse_args()
    scrape_all_operators(periodo=args.periodo, operators=args.operators)
