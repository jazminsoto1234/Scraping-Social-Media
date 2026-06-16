"""
Descarga TODOS los comentarios de un video de TikTok.
Navega directo al video, hace scroll automatico en el panel de comentarios
y expande replies. Guarda todo en video_{ID}_comments.json dentro de test/.
"""
from playwright.sync_api import sync_playwright
import json
import re
import time
import random
import logging
from datetime import datetime

STORAGE_FILE = "tiktok_storage_state.json"
VIDEO_URL = "https://www.tiktok.com/@eleduoficial23/video/7643490547780390164"
#https://www.tiktok.com/@cibercrimen_pnp_/video/7642197317679942933"

# Archivo de salida — se escribe de forma incremental en cada batch
m_url = re.match(r'https?://(?:www\.)?tiktok\.com/@([^/]+)/video/(\d+)', VIDEO_URL)
VIDEO_ID = m_url.group(2) if m_url else 'unknown'
OUTPUT_FILE = f"video_{VIDEO_ID}_comments.json"

logging.basicConfig(
    filename='cuarto_video_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


def guardar_incremental(all_comments, all_replies):
    """Escribe el JSON de salida con lo capturado hasta ahora."""
    video_meta = {
        'id': VIDEO_ID,
        'autor': re.match(r'https?://(?:www\.)?tiktok\.com/@([^/]+)/', VIDEO_URL).group(1),
        'url': VIDEO_URL,
    }
    comentarios_simplificados = []
    for comment in all_comments:
        user = comment.get('user') or {}
        comment_id = str(comment.get('cid', ''))
        respuestas_simples = []
        for reply in (comment.get('reply_comment') or []):
            ru = reply.get('user') or {}
            ts = reply.get('create_time', 0)
            respuestas_simples.append({
                'id': str(reply.get('cid', '')),
                'usuario_id': ru.get('unique_id', 'unknown'),
                'nickname': ru.get('nickname', ''),
                'comentario': reply.get('text', ''),
                'likes': reply.get('digg_count', 0),
                'fecha': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else '',
            })
        for reply in (all_replies.get(comment_id) or []):
            ru = reply.get('user') or {}
            ts = reply.get('create_time', 0)
            respuestas_simples.append({
                'id': str(reply.get('cid', '')),
                'usuario_id': ru.get('unique_id', 'unknown'),
                'nickname': ru.get('nickname', ''),
                'comentario': reply.get('text', ''),
                'likes': reply.get('digg_count', 0),
                'fecha': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else '',
            })
        ts = comment.get('create_time', 0)
        comentarios_simplificados.append({
            'id': comment_id,
            'usuario_id': user.get('unique_id', 'unknown'),
            'nickname': user.get('nickname', ''),
            'comentario': comment.get('text', ''),
            'likes': comment.get('digg_count', 0),
            'fecha': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else '',
            'num_respuestas': comment.get('reply_comment_total', 0),
            'respuestas': respuestas_simples,
        })
    output = {
        'video': video_meta,
        'total_comentarios': len(comentarios_simplificados),
        'total_respuestas': sum(len(c['respuestas']) for c in comentarios_simplificados),
        'total_items': len(comentarios_simplificados) + sum(len(c['respuestas']) for c in comentarios_simplificados),
        'comentarios': comentarios_simplificados,
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.debug(f"Guardado incremental: {output['total_items']} items -> {OUTPUT_FILE}")


def scraper_main():
    """Obtiene comentarios del video via API directa interceptando parametros reales."""

    all_comments = []
    all_replies = {}
    seen_comment_ids = set()
    seen_reply_ids = set()

    def handle_response(response):
        url = response.url
        guardado = False

        if 'api/comment/list' in url and 'reply' not in url:
            try:
                data = response.json()
                batch = data.get('comments') or []
                nuevos = 0
                for c in batch:
                    cid = str(c.get('cid', ''))
                    if cid and cid not in seen_comment_ids:
                        seen_comment_ids.add(cid)
                        all_comments.append(c)
                        nuevos += 1
                if nuevos:
                    logging.info(f"+{nuevos} comentarios, total: {len(all_comments)}")
                    guardado = True
            except Exception as e:
                logging.error(f"Error parseando comentarios: {e}")

        if '/api/comment/list/reply/' in url:
            try:
                data = response.json()
                replies = data.get('comments') or []
                if replies:
                    match = re.search(r'comment_id=(\d+)', url)
                    if match:
                        parent_id = match.group(1)
                        if parent_id not in all_replies:
                            all_replies[parent_id] = []
                        nuevos_r = 0
                        for r in replies:
                            rid = str(r.get('cid', ''))
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
                guardar_incremental(all_comments, all_replies)
            except Exception as e:
                logging.error(f"Error guardado incremental: {e}")

    with sync_playwright() as p:
        print("Conectando al Chrome real via CDP (localhost:9222)...")
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"❌ No se pudo conectar: {e}")
            print("   Asegurate de tener Chrome abierto con --remote-debugging-port=9222")
            return None, [], {}

        context = browser.contexts[0]

        # Maximizar ventana para que el panel de comentarios quede visible y a la derecha
        if context.pages:
            context.pages[0].evaluate("() => { window.moveTo(0,0); window.resizeTo(screen.availWidth, screen.availHeight); }")

        # Usar pestaña existente con el video o abrir una nueva
        page = None
        for pg in context.pages:
            if VIDEO_ID in pg.url:
                page = pg
                print(f"  Usando pestaña existente: {pg.url[:80]}")
                break

        if page is None:
            page = context.new_page()
            print(f"  Navegando al video...")
            page.goto(VIDEO_URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(4000)

        # Si TikTok redirigió a login, esperar hasta 60s a que el usuario inicie sesión
        if 'login' in page.url or 'passport' in page.url:
            print("  TikTok pide login — inicia sesión en el navegador (tienes 60s)...")
            for _ in range(60):
                page.wait_for_timeout(1000)
                if 'login' not in page.url and 'passport' not in page.url:
                    print("  Login detectado, continuando...")
                    page.goto(VIDEO_URL, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(4000)
                    break
            else:
                print("  Timeout esperando login — abortando")
                return None, [], {}

        page.on('response', handle_response)

        page.wait_for_timeout(3000)

        # Click en pestaña Comentarios del video (data-e2e="comment-tab" o aria-label).
        # En la vista de video con columna derecha los comentarios ya están visibles
        # sin tab — el click puede devolver None y es normal en ese layout.
        resultado_tab = page.evaluate("""
            () => {
                // Buscar tab de comentarios del video, no del inbox
                const candidates = [
                    ...document.querySelectorAll('[data-e2e="comment-tab"]'),
                    ...document.querySelectorAll('[data-e2e="comments-tab"]'),
                    ...document.querySelectorAll('[data-e2e="browse-comment-tab"]'),
                ];
                for (const el of candidates) {
                    el.click();
                    return 'data-e2e:' + el.getAttribute('data-e2e');
                }
                // fallback: buscar tab visible con texto exacto (no el del inbox)
                for (const el of document.querySelectorAll('[role="tab"]')) {
                    const txt = el.textContent.trim();
                    if ((txt === 'Comentarios' || txt === 'Comments') && el.offsetParent !== null) {
                        el.click();
                        return 'role-tab-visible:' + txt;
                    }
                }
                // fallback: click en el icono de burbuja de comentarios (vista columna derecha)
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
        print(f"  Tab click: {resultado_tab}")

        # Esperar a que el panel scrollable aparezca en el DOM (máx 8s)
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
                print(f"  Panel detectado tras {(_w+1)*0.5:.1f}s")
                break
        if not panel_listo:
            print("  AVISO: panel scrollable no detectado, continuando igual")

        # Helper: obtener centro del panel scrollable (DivColumnListContain u otro).
        # Filtra por bounding rect: r.y > 150 excluye el contenedor externo que incluye
        # el header fijo, y r.height > 400 garantiza que es el área de lista real.
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

        # Helper: scroll humano (mouse wheel real via CDP sobre el panel).
        # Usa coords pre-calibradas _px/_py del closure para no recalcular en cada wheel.
        # Devuelve (scrollTop_antes, scrollTop_despues).
        # Nunca usa scrollTop += como fallback — si wheel no mueve, reposiciona y reintenta.
        def scroll_panel(delta, px, py):
            antes = _get_scroll_top()
            if antes is None:
                return None, None
            page.mouse.move(px, py)
            page.mouse.wheel(0, delta)
            page.wait_for_timeout(60)
            despues = _get_scroll_top()
            # Si el wheel no movió nada: reposicionar y reintentar hasta 2 veces
            for _retry in range(2):
                if despues is not None and despues != antes:
                    break
                logging.warning(f"  Wheel no movió (intento {_retry+1}), reposicionando...")
                page.mouse.move(px, py)
                page.wait_for_timeout(80)
                page.mouse.wheel(0, delta)
                page.wait_for_timeout(60)
                despues = _get_scroll_top()
            if despues is not None and despues == antes:
                logging.warning("  Wheel sin efecto tras 2 reintentos, continuando sin scrollTop+=")
            return antes, despues

        # Calibrar coordenadas UNA sola vez antes del loop principal
        _c = centro_panel()
        if _c:
            _px, _py = _c['x'], _c['y']
        else:
            _px, _py = 0.0, 0.0
        print(f"  Panel centro: {_c}")
        print(f"  Panel calibrado: x={_px:.0f} y={_py:.0f}")

        # ── PASADA 1+2: scroll continuo para cargar todos los comentarios padre ──
        print("Cargando comentarios padre via scroll...")
        sin_cambios = 0
        prev_count = 0
        grupos_desde_ultimo_cambio = 0
        grupo_num = 0

        while True:
            grupo_num += 1
            pasos = random.randint(5, 10)
            # Problema 4: mover el cursor explícitamente al panel antes del grupo
            page.mouse.move(_px, _py)
            for _ in range(pasos):
                delta = random.randint(150, 280)
                # Problema 3: delay más largo entre wheels individuales
                scroll_panel(delta, _px, _py)
                page.wait_for_timeout(random.randint(200, 450))

            page.wait_for_timeout(random.randint(400, 800))

            # Problema 3: pausa larga cada 5-8 grupos (simula lectura humana)
            if grupo_num % random.randint(5, 8) == 0:
                pausa_larga = random.randint(1500, 3000)
                logging.debug(f"  Pausa larga ({pausa_larga}ms) en grupo {grupo_num}")
                page.wait_for_timeout(pausa_larga)

            if len(all_comments) > prev_count:
                sin_cambios = 0
                grupos_desde_ultimo_cambio = 0
                print(f"  comentarios: {len(all_comments)}")
                guardar_incremental(all_comments, all_replies)
                prev_count = len(all_comments)
            else:
                sin_cambios += 1
                grupos_desde_ultimo_cambio += 1

                # Problema 5: recovery intermedio al llegar a 5 sin cambios
                if sin_cambios == 5:
                    print(f"  Sin cambios x5 — recalibrando panel y reintentando...")
                    _c2 = centro_panel()
                    if _c2:
                        _px, _py = _c2['x'], _c2['y']
                        print(f"  Panel recalibrado: x={_px:.0f} y={_py:.0f}")
                    page.mouse.move(_px, _py)
                    for _ in range(3):
                        page.mouse.move(_px, _py)
                        for _ in range(random.randint(5, 8)):
                            scroll_panel(random.randint(150, 280), _px, _py)
                            page.wait_for_timeout(random.randint(200, 450))
                        page.wait_for_timeout(random.randint(400, 800))
                    # Si los reintentos trajeron nuevos comentarios, resetear contador
                    if len(all_comments) > prev_count:
                        sin_cambios = 0
                        grupos_desde_ultimo_cambio = 0
                        prev_count = len(all_comments)
                        print(f"  Recovery exitoso: {len(all_comments)} comentarios")

                if sin_cambios >= 10:
                    print(f"  comentarios sin cambio — scroll terminado ({len(all_comments)})")
                    break

        print(f"Comentarios padre: {len(all_comments)}")

        # ── PASADA 3: expandir replies clickeando botones visibles ──
        print("Expandiendo replies (scroll + click)...")
        clicks_totales = 0
        sin_cambios = 0
        prev_replies = 0

        reply_selectors = [
            'text=/ver \\d+ respuesta/i',
            'text=/ver \\d+ repl/i',
            'text=/\\d+ respuesta/i',
            'text=/\\d+ repl/i',
            'text=/Ver.*más/i',
            'text=/View.*more/i',
        ]

        def expandir_replies_visibles():
            clicked_local = 0
            for _ in range(6):
                nuevos_clicks = 0
                for selector in reply_selectors:
                    try:
                        for elem in page.locator(selector).all():
                            try:
                                if elem.is_visible(timeout=300):
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

        # Volver al top del contenedor (wheel negativo + scrollTop=0 explícito)
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

        TIMEOUT_INACTIVIDAD_REPLIES = 3 * 60  # 3 min sin que el JSON crezca → salir
        ultimo_progreso_replies = time.time()

        while True:
            if (time.time() - ultimo_progreso_replies) > TIMEOUT_INACTIVIDAD_REPLIES:
                print(f"  replies: JSON sin crecer 3 min — saliendo ({sum(len(r) for r in all_replies.values())} replies)")
                break

            page.mouse.move(_px, _py)
            for _ in range(random.randint(4, 7)):
                delta = random.randint(120, 220)
                scroll_panel(delta, _px, _py)
                page.wait_for_timeout(random.randint(200, 450))

            clicks = expandir_replies_visibles()
            if clicks > 0:
                clicks_totales += clicks

            page.wait_for_timeout(random.randint(600, 1200))

            current_replies = sum(len(r) for r in all_replies.values())
            if current_replies > prev_replies:
                ultimo_progreso_replies = time.time()
                print(f"  replies: {current_replies}")
                guardar_incremental(all_comments, all_replies)
            prev_replies = current_replies

        total_replies_capturadas = sum(len(r) for r in all_replies.values())
        print(f"Terminado: {len(all_comments)} comentarios, {total_replies_capturadas} replies")

        # Guardado final explícito — las replies capturadas por click pueden no
        # haber disparado handle_response si TikTok las sirvió desde caché.
        guardar_incremental(all_comments, all_replies)
        print(f"  Guardado final: {OUTPUT_FILE}")

        browser.close()

    m = re.match(r'https?://(?:www\.)?tiktok\.com/@([^/]+)/video/(\d+)', VIDEO_URL)
    video_meta = {
        'id': m.group(2) if m else '',
        'autor': m.group(1) if m else '',
        'url': VIDEO_URL,
    }
    return video_meta, all_comments, all_replies


if __name__ == "__main__":
    inicio_total = time.time()
    _, comentarios, _ = scraper_main()

    if not comentarios:
        print("\nAVISO: No se capturaron comentarios")
    else:
        # El archivo ya fue guardado incrementalmente durante la ejecucion
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            output = json.load(f)
        print(f"\nResultados:")
        print(f"  Comentarios principales : {output['total_comentarios']}")
        print(f"  Respuestas totales      : {output['total_respuestas']}")
        print(f"  Total items             : {output['total_items']}")
        print(f"  Archivo                 : {OUTPUT_FILE}")

    tiempo_total = time.time() - inicio_total
    print(f"\nTiempo de ejecucion: {tiempo_total:.1f}s")
