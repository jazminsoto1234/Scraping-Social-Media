"""
Scraper de Facebook usando Playwright + interceptacion GraphQL.
Paso 2 del pipeline: consume el JSON generado por profile_post_lister.py
y extrae comentarios de cada post.

Flujo por operadora:
  1. Lee outputs/listas/{periodo}/{operador}_posts.json
  2. Por cada entrada { "url": ..., "fecha": ... }:
       - Navega directamente a la URL del post
       - Abre el modal de comentarios
       - Extrae comentarios via GraphQL o DOM
       - Guarda en outputs/{operador}/mes_{periodo}/facebook/post_{ID}.json

Uso:
    python src/scrapers/facebook/scraper.py --periodo 202506
    python src/scrapers/facebook/scraper.py --periodo 202506 --operators entel claro
"""
import re
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from playwright.sync_api import sync_playwright, Page

logging.basicConfig(
    filename="facebook_scraper.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point principal
# ---------------------------------------------------------------------------

def scrape_all_operators(periodo: str, operators: list = None):
    config = _cargar_config()
    fb_cfg = config["facebook"]

    todos = config.get("operators_facebook", {})
    if operators:
        todos = {k: v for k, v in todos.items() if k in operators}

    if not todos:
        print("ERROR: No hay operadores configurados (revisa config.yaml o el filtro pasado)")
        return

    resultados = []

    with sync_playwright() as p:
        browser, page = _build_browser(p)
        try:
            for operador in todos:
                resultado = _scrape_operator(page, operador, periodo, fb_cfg)
                resultados.append(resultado)
        finally:
            browser.close()

    print(f"\n{'='*60}")
    print(f"RESUMEN — periodo {periodo}")
    print(f"{'='*60}")
    print(f"  {'Operador':<12} {'Total':>7} {'Guardados':>10} {'Errores':>8}")
    print(f"  {'-'*40}")
    for r in resultados:
        print(f"  {r['operador']:<12} {r['total']:>7} {r['guardados']:>10} {r['errores']:>8}")


# ---------------------------------------------------------------------------
# Scraping por operadora
# ---------------------------------------------------------------------------

def _cargar_lista_posts(operador: str, periodo: str) -> list:
    lista_path = Path("outputs") / "listas" / periodo / f"{operador}_posts.json"
    if not lista_path.exists():
        print(f"  ERROR: no existe {lista_path}")
        print(f"  Ejecuta primero: python src/scrapers/facebook/profile_post_lister.py --periodo {periodo}")
        return []
    with open(lista_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    posts = data.get("posts", [])
    print(f"  Lista cargada: {lista_path}  ({len(posts)} posts)")
    return posts


def _extraer_post_id(url: str) -> str:
    m = re.search(r"(pfbid[\w]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/posts/(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/reel/(\d+)", url)
    if m:
        return f"reel_{m.group(1)}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", url[-40:])


def _scrape_operator(page: Page, operador: str, periodo: str, fb_cfg: dict) -> dict:
    output_dir = Path("outputs") / operador / f"mes_{periodo}" / "facebook"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Operador: {operador}  |  periodo: {periodo}")
    print(f"{'='*60}")

    posts_lista = _cargar_lista_posts(operador, periodo)
    if not posts_lista:
        return {"operador": operador, "total": 0, "guardados": 0, "errores": 0}

    guardados = 0
    errores = 0

    for idx, entry in enumerate(posts_lista):
        url = entry.get("url", "")
        fecha_conocida = entry.get("fecha")

        if not url:
            logger.warning("Entrada %d sin URL, saltando", idx)
            continue

        post_id = _extraer_post_id(url)
        output_path = output_dir / f"post_{post_id}.json"

        if output_path.exists():
            print(f"  [{idx+1}/{len(posts_lista)}] Ya existe {output_path.name}, saltando.")
            guardados += 1
            continue

        print(f"\n  [{idx+1}/{len(posts_lista)}] {url[:70]}")
        logger.info("Procesando post %d/%d: %s", idx + 1, len(posts_lista), url)

        post_result = _process_post(page, url, fecha_conocida, fb_cfg, output_path, operador)

        if not post_result:
            errores += 1
            logger.warning("No se pudo procesar %s", url)
            continue

        _guardar_post(post_result, output_path, operador)
        guardados += 1
        print(f"    Guardado: {output_path.name}  ({len(post_result['comments'])} comentarios)")

    return {"operador": operador, "total": len(posts_lista), "guardados": guardados, "errores": errores}


def _guardar_acumulado_incremental(acumulado: dict, output_path: Path, operador: str, url: str):
    """Escribe al disco el estado actual del acumulado durante el scroll."""
    try:
        comments = _construir_jerarquia(acumulado)
        data = {
            "operador": operador,
            "post": {
                "url": url,
                "fecha": None,
                "scrapeado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "total_comentarios": len(comments),
            "total_respuestas": sum(len(c.get("respuestas", [])) for c in comments),
            "comentarios": comments,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Guardado incremental: %d comentarios → %s", len(comments), output_path.name)
    except Exception as e:
        logger.debug("Error en guardado incremental: %s", e)


def _guardar_post(post_result: dict, output_path: Path, operador: str):
    data = {
        "operador": operador,
        "post": post_result["post"],
        "total_comentarios": len(post_result["comments"]),
        "total_respuestas": sum(len(c.get("respuestas", [])) for c in post_result["comments"]),
        "comentarios": post_result["comments"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Config y browser
# ---------------------------------------------------------------------------

def _cargar_config() -> dict:
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_browser(p):
    print("Conectando al Chrome real via CDP (localhost:9222)...")
    try:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
    except Exception as e:
        print(f"No se pudo conectar: {e}")
        print("  Asegurate de tener Chrome abierto con --remote-debugging-port=9222")
        raise
    context = browser.contexts[0]
    page = context.new_page()
    return browser, page


# ---------------------------------------------------------------------------
# Procesamiento de un post: navega a la URL, abre modal, extrae comentarios
# ---------------------------------------------------------------------------

def _es_reel(url: str) -> bool:
    return "/reel/" in url


def _process_post(page: Page, url: str, fecha_conocida: Optional[str], fb_cfg: dict,
                  output_path: Path = None, operador: str = None) -> Optional[dict]:
    graphql_nodes = []
    capturar_graphql = False

    def handle_response(response):
        nonlocal capturar_graphql
        if "/api/graphql/" in response.url and capturar_graphql:
            try:
                data = response.json()
                graphql_nodes.extend(_extract_nodes_from_graphql(data))
            except Exception:
                pass

    page.on("response", handle_response)

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)

        capturar_graphql = True

        if _es_reel(url):
            comments, fecha_str = _extraer_comentarios_reel(page, fb_cfg, output_path, operador)
        else:
            comments, fecha_str = _extraer_comentarios_post(page, fb_cfg, output_path, operador)

        if fecha_str is None and fecha_conocida:
            fecha_str = fecha_conocida

        post_data = {
            "url": url,
            "fecha": fecha_str,
            "scrapeado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if graphql_nodes:
            comments = _process_graphql_nodes(graphql_nodes)
            logger.info("Comentarios via GraphQL: %d", len(comments))
        else:
            logger.info("Comentarios via DOM: %d", len(comments))

    except Exception as e:
        logger.error("Error procesando %s: %s", url, e)
        page.remove_listener("response", handle_response)
        return None
    finally:
        page.remove_listener("response", handle_response)

    return {"post": post_data, "comments": comments}


def _extraer_comentarios_reel(page: Page, fb_cfg: dict,
                              output_path: Path = None, operador: str = None) -> tuple[list, Optional[str]]:
    """
    Para reels: clickea el icono de comentar (boton con aria-label que contiene
    'comentario' o 'comment'), espera el panel lateral y scrollea dentro de el.
    """
    # El icono de comentar en reels no tiene texto visible sino aria-label.
    # Se prueban selectores genericos por idioma (no se hardcodea el numero de
    # comentarios, que varia por reel).
    selectores_boton = [
        '[role="button"][aria-label*="omenta" i]',   # "Comentar", "Comentarios"
        '[role="button"][aria-label*="comment" i]',  # "Comment", "Comments"
        '[aria-label*="omenta" i]',                  # mismo, sin exigir role=button
        '[aria-label*="comment" i]',
    ]
    boton = None
    for sel in selectores_boton:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.is_visible(timeout=2000):
                boton = loc
                break
        except Exception:
            pass

    # Fallback: boton con un icono de comentar y un contador numerico (cualquier
    # numero, no uno fijo). Se localiza por JS y se devuelven coordenadas para click real.
    if boton is None:
        try:
            coord = page.evaluate("""
                () => {
                    const re = /\\d+/;
                    const botones = Array.from(document.querySelectorAll('[role="button"]'));
                    for (const b of botones) {
                        const al = (b.getAttribute('aria-label') || '').toLowerCase();
                        const txt = (b.textContent || '').trim();
                        // icono de comentar: aria-label o un svg de comentario + numero visible
                        if ((/coment|comment/.test(al)) ||
                            (b.querySelector('svg, i[data-visualcompletion]') && re.test(txt) && txt.length < 12)) {
                            const r = b.getBoundingClientRect();
                            if (r.width > 0 && r.height > 0)
                                return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
                        }
                    }
                    return null;
                }
            """)
        except Exception:
            coord = None
        if coord:
            page.mouse.click(coord["x"], coord["y"])
            page.wait_for_timeout(2000)
            logger.debug("Reel: boton de comentar abierto via fallback JS")
            boton = "fallback"  # marca que ya se clickeo, no volver a clickear abajo

    if boton is None:
        logger.warning("Reel: no se encontro boton de comentar en %s", page.url)
        return [], None

    if boton != "fallback":
        boton.click()
        page.wait_for_timeout(2000)

    # El panel de comentarios del reel suele ser [role="complementary"]
    # (a veces dialog o aside). Esperar a que aparezca CON comentarios dentro.
    contenedor = None
    for _ in range(10):  # hasta ~5s esperando que carguen los comentarios
        for sel in ['[role="complementary"]', '[role="dialog"]', 'aside']:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.locator(_SELECTOR_COMENTARIOS).count() > 0:
                contenedor = loc
                break
        if contenedor is not None:
            break
        page.wait_for_timeout(500)

    if contenedor is None:
        # Si no hay panel separado, usar la pagina entera
        logger.debug("Reel: sin panel con comentarios, scrolleando pagina")
        contenedor = page

    fecha_str, _ = _extract_post_date_from_container(contenedor)
    _switch_to_all_comments(page, contenedor)

    acumulado: dict = {}

    # FASE A: agotar comentarios principales hasta que el acumulado deje de crecer
    _agotar_scroll(
        page, contenedor, fb_cfg, acumulado,
        fase="comentarios-reel",
        scroll_fn=_scroll_fn_reel(page, contenedor),
        paginar_fn=lambda: _click_ver_mas_comentarios(page, contenedor),
        output_path=output_path, operador=operador,
    )

    # FASE B: agotar respuestas hasta que el acumulado deje de crecer
    _agotar_replies(page, contenedor, fb_cfg, acumulado, output_path, operador)

    return _construir_jerarquia(acumulado), fecha_str


def _extraer_comentarios_post(page: Page, fb_cfg: dict,
                              output_path: Path = None, operador: str = None) -> tuple[list, Optional[str]]:
    """
    Para posts normales: al navegar a la URL se abre un modal/dialog con la imagen
    arriba y los comentarios abajo. Hay que scrollear DENTRO del modal, no en la pagina.
    """
    # Esperar que aparezca el modal con comentarios dentro
    contenedor = None
    for _ in range(20):  # hasta ~10s
        for sel in ['[role="dialog"]', '[role="main"]']:
            loc = page.locator(sel).first
            try:
                if loc.count() > 0 and loc.locator(_SELECTOR_COMENTARIOS).count() > 0:
                    contenedor = loc
                    break
            except Exception:
                pass
        if contenedor is not None:
            break
        page.wait_for_timeout(500)

    if contenedor is None:
        # fallback: no hay modal, scrollear la pagina entera
        logger.debug("Post: sin modal con comentarios, usando pagina entera")
        contenedor = page

    fecha_str, _ = _extract_post_date_from_container(contenedor)

    # "Todos los comentarios" opera dentro del modal
    _switch_to_all_comments(page, contenedor)

    acumulado: dict = {}

    # FASE A: agotar comentarios principales hasta que el acumulado deje de crecer
    # El scroll es dentro del modal (mismo mecanismo que reel: busca div scrolleable)
    _agotar_scroll(
        page, contenedor, fb_cfg, acumulado,
        fase="comentarios-post",
        scroll_fn=_scroll_fn_reel(page, contenedor),  # reel y post usan el mismo scroll de div interno
        paginar_fn=lambda: _click_ver_mas_comentarios(page, contenedor),
        output_path=output_path, operador=operador,
    )

    # FASE B: agotar respuestas hasta que el acumulado deje de crecer
    _agotar_replies(page, contenedor, fb_cfg, acumulado, output_path, operador)

    return _construir_jerarquia(acumulado), fecha_str


def _extract_post_date_from_container(container) -> tuple[Optional[str], Optional[datetime]]:
    """Extrae la fecha del post desde cualquier contenedor (modal, page, aside)."""
    selectores = [
        "a[href*='/posts/'] span[id]",
        "a[href*='/posts/']",
        "a[href*='/reel/']",
        "abbr[data-utime]",
        "a[role='link'] > span",
    ]
    for selector in selectores:
        try:
            elems = container.locator(selector).all()
            for elem in elems:
                for attr in ["title", "aria-label", "data-utime"]:
                    val = elem.get_attribute(attr) or ""
                    if val and re.search(r"\d{4}", val):
                        fecha_dt = _parse_fb_date(val)
                        if fecha_dt:
                            logger.debug("Fecha extraida: %s → %s", val, fecha_dt.date())
                            return val, fecha_dt
        except Exception:
            continue
    logger.debug("No se pudo extraer fecha del contenedor")
    return None, None


def _parse_fb_date(text: str) -> Optional[datetime]:
    patrones = [
        r"(\d{1,2}) de (\w+) de (\d{4})",
        r"(\w+) (\d{1,2}),?\s*(\d{4})",
        r"(\d{4})-(\d{2})-(\d{2})",
    ]
    meses_es = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    }
    meses_en = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for patron in patrones:
        m = re.search(patron, text, re.IGNORECASE)
        if not m:
            continue
        try:
            if "-" in patron:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            g1, g2, anio = m.group(1).lower(), m.group(2).lower(), int(m.group(3))
            if g1.isdigit():
                dia = int(g1)
                mes = meses_es.get(g2) or meses_en.get(g2)
            else:
                mes = meses_es.get(g1) or meses_en.get(g1)
                dia = int(g2.rstrip(","))
            if mes:
                return datetime(anio, mes, dia)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Helpers de interaccion con el modal
# ---------------------------------------------------------------------------

def _switch_to_all_comments(page: Page, container):
    """
    Abre el dropdown de orden de comentarios y elige 'Todos los comentarios'.
    Funciona tanto en posts (dropdown visible en pagina) como en reels (dentro de panel).
    El dropdown puede estar ya abierto mostrando las opciones, o cerrado mostrando
    el filtro activo ('Mas relevantes', 'Más recientes', 'Top comments', etc.).
    """
    try:
        # Paso 1: si el dropdown esta cerrado, abrirlo clickeando el filtro activo
        trigger_patterns = [
            r"Más relevantes|Mas relevantes|Most relevant",
            r"Más recientes|Mas recientes|Most recent",
            r"Top comments",
            r"Todos los comentarios|All comments",  # ya seleccionado — no hace falta cambiar
        ]
        for pattern in trigger_patterns:
            try:
                loc = container.locator(f"text=/{pattern}/i").first
                if loc.count() > 0 and loc.is_visible(timeout=1500):
                    # Si ya dice "Todos los comentarios" no hay que hacer nada
                    texto_actual = loc.inner_text(timeout=1000).strip().lower()
                    if "todos" in texto_actual or "all comments" in texto_actual:
                        logger.debug("Ya esta en Todos los comentarios")
                        return
                    loc.click()
                    page.wait_for_timeout(1000)
                    break
            except Exception:
                continue

        # Paso 2: elegir "Todos los comentarios" del dropdown abierto
        for opcion in ["Todos los comentarios", "All comments"]:
            try:
                btn = page.locator(f'text="{opcion}"').first
                if btn.count() > 0 and btn.is_visible(timeout=2000):
                    btn.click()
                    page.wait_for_timeout(2000)
                    logger.info("Cambiado a: %s", opcion)
                    return
            except Exception:
                continue

    except Exception as e:
        logger.debug("Error cambiando orden de comentarios: %s", e)


def _agotar_scroll(page: Page, container, fb_cfg: dict, acumulado: dict,
                   fase: str, scroll_fn, paginar_fn,
                   output_path: Path = None, operador: str = None):
    """
    Bucle generico: scrollea y pagina hasta que el acumulado DEJA DE CRECER.

    Parada principal: si tras `fb_max_vueltas_sin_crecer` vueltas seguidas no entro
    ningun comentario/respuesta nuevo al acumulado, se cierra la fase (ya no se puede
    hacer nada mas). Los topes `scroll_timeout_total_s` y `max_scroll_rounds` solo son
    red de seguridad para no colgarse.

    Parametrizado por:
      - scroll_fn(): mueve la rueda / scrollTo (difiere entre reel y post).
      - paginar_fn(): dispara la accion de paginacion ("Ver mas comentarios" o expandir
                      respuestas). Debe devolver algo truthy si hizo click, o None.
    La unica senal que importa es si el ACUMULADO crecio, no si hubo clicks.
    """
    timeout = max(fb_cfg.get("scroll_timeout_ms", 300), 1200)
    max_vueltas_sin_crecer = fb_cfg.get("fb_max_vueltas_sin_crecer", 5)
    max_rounds = fb_cfg.get("max_scroll_rounds", 60)
    timeout_total_s = fb_cfg.get("scroll_timeout_total_s", 300)
    sin_crecer = 0
    t_inicio = datetime.now()

    _recolectar_visibles(container, acumulado)

    for ronda in range(max_rounds):
        n_antes = len(acumulado)

        try:
            scroll_fn()
        except Exception:
            pass
        try:
            paginar_fn()
        except Exception:
            pass

        page.wait_for_timeout(timeout)
        _recolectar_visibles(container, acumulado)
        crecio = len(acumulado) - n_antes

        if crecio > 0:
            sin_crecer = 0
            logger.debug("[%s] ronda %d: +%d nuevos (total %d)",
                         fase, ronda + 1, crecio, len(acumulado))
            # Guardado incremental para no perder progreso ante crash
            if output_path and operador:
                _guardar_acumulado_incremental(acumulado, output_path, operador, page.url)
        else:
            sin_crecer += 1
            logger.debug("[%s] ronda %d: sin crecer (%d/%d)",
                         fase, ronda + 1, sin_crecer, max_vueltas_sin_crecer)

        # PARADA PRINCIPAL: dejo de crecer
        if sin_crecer >= max_vueltas_sin_crecer:
            logger.debug("[%s]: cerrado por %d vueltas sin crecer", fase, max_vueltas_sin_crecer)
            break
        # red de seguridad
        if int((datetime.now() - t_inicio).total_seconds()) >= timeout_total_s:
            logger.debug("[%s]: parado por timeout total (%ds) — posible incompleto", fase, timeout_total_s)
            break


def _scroll_fn_reel(page: Page, container):
    """
    Devuelve una funcion que scrollea el div de comentarios de un reel.

    El panel de comentarios del reel ([role="complementary"]) NO contiene el div
    scrolleable: el div con overflow real cuelga de [role="main"], fuera del panel.
    Por eso el JS busca el div scrolleable en TODO el documento (el de mayor
    scrollHeight-clientHeight con overflowY auto/scroll), no dentro del panel.
    """
    def _scroll():
        div_info = None
        try:
            div_info = page.evaluate("""
                () => {
                    const tieneComment = el => el.querySelector(
                        '[aria-label^="Comentario de"], [aria-label^="Respuesta de"], ' +
                        '[aria-label^="Comment by"], [aria-label^="Reply by"]') !== null;
                    let mejor = null, maxDelta = 0;
                    document.querySelectorAll('*').forEach(el => {
                        const delta = el.scrollHeight - el.clientHeight;
                        const oy = getComputedStyle(el).overflowY;
                        if (delta > 200 && (oy==='auto'||oy==='scroll') && tieneComment(el))
                            if (delta > maxDelta) { maxDelta = delta; mejor = el; }
                    });
                    if (!mejor) return null;
                    const r = mejor.getBoundingClientRect();
                    return { cx: r.x + r.width / 2, cy: r.y + r.height / 2 };
                }
            """)
        except Exception:
            pass

        # Siempre scrollear: al fondo significa que Facebook aun no cargo el
        # siguiente lote; la rueda dispara el lazy-load y hace crecer scrollHeight.
        if div_info:
            page.mouse.move(div_info["cx"], div_info["cy"])
            for _ in range(3):
                page.mouse.wheel(0, 500)
                page.wait_for_timeout(150)
        else:
            comentarios = container.locator(_SELECTOR_COMENTARIOS).all()
            if comentarios:
                box = comentarios[-1].bounding_box()
                ancho = page.viewport_size["width"] if page.viewport_size else 1920
                if box and (box["x"] + box["width"] / 2) > ancho / 2:
                    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    page.mouse.wheel(0, 1000)

    return _scroll


def _scroll_fn_post(page: Page):
    """Devuelve una funcion que scrollea la pagina entera al fondo (posts normales)."""
    def _scroll():
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    return _scroll


def _click_ver_mas_comentarios(page: Page, container) -> int:
    """
    Clickea botones de paginacion de comentarios via coordenadas reales (mouse.click),
    no con target.click() sintetico que React puede ignorar.
    Devuelve cuantos clicks hizo.
    """
    regex = (
        r"Ver más comentarios|Ver mas comentarios|View more comments|"
        r"Ver \d+ comentarios?|View \d+ comments?|"
        r"Ver las \d+ respuestas|Ver \d+ respuestas?|Ver \d+ respuesta|"
        r"View \d+ repl(?:y|ies)|View \d+ more repl(?:y|ies)"
    )
    # JS solo localiza coordenadas; el click real lo hace Playwright
    try:
        coordenadas = page.evaluate(
            """
            (regexStr) => {
                const re = new RegExp(regexStr, 'i');
                // Solo buscar "Ver más comentarios" (paginacion principal); las respuestas
                // las expande _agotar_replies (Fase B), despues de agotar los comentarios.
                const esPaginacion = t => /Ver m[aá]s comentarios|View more comments|Ver \\d+ comentarios?|View \\d+ comments?/.test(t);
                const resultados = [];
                const vistos = new Set();
                document.querySelectorAll('span, div').forEach(el => {
                    if (el.childElementCount > 0) return;
                    const t = (el.textContent || '').trim();
                    if (t.length < 60 && re.test(t)) {
                        let target = el;
                        for (let i = 0; i < 8 && target; i++) {
                            const role = target.getAttribute && target.getAttribute('role');
                            const cur = getComputedStyle(target).cursor;
                            if (role === 'button' || target.onclick || cur === 'pointer') {
                                break;
                            }
                            target = target.parentElement;
                        }
                        target = target || el;
                        if (vistos.has(target)) return;
                        vistos.add(target);
                        const rect = target.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) return;
                        target.scrollIntoView({block: 'center'});
                        resultados.push({
                            x: rect.x + rect.width / 2,
                            y: rect.y + rect.height / 2,
                            esPaginacion: esPaginacion(t),
                        });
                    }
                });
                // Si hay boton de paginacion principal, solo clickear ese
                const pagMain = resultados.filter(r => r.esPaginacion);
                return pagMain.length > 0 ? pagMain.slice(0, 1) : resultados.slice(0, 5);
            }
            """,
            regex,
        )
    except Exception as e:
        logger.debug("Error en _click_ver_mas_comentarios JS: %s", e)
        coordenadas = []

    clicks = 0
    for coord in (coordenadas or []):
        try:
            page.mouse.click(coord["x"], coord["y"])
            clicks += 1
        except Exception as e:
            logger.debug("Error clickeando coordenadas %s: %s", coord, e)

    if clicks:
        page.wait_for_timeout(1500)
        logger.debug("Ver más comentarios/respuestas: %d clicks reales", clicks)
    return clicks


def _agotar_replies(page: Page, container, fb_cfg: dict, acumulado: dict,
                    output_path: Path = None, operador: str = None):
    """
    FASE B: expande las respuestas ("Ver N respuestas") hasta que el acumulado DEJA
    DE CRECER, con la MISMA semantica de cierre que la Fase A de comentarios.

    Tras cada click de expansion se recolectan los elementos visibles. Si el acumulado
    crecio, se resetea el contador y se guarda incremental; si no, se cuenta una vuelta
    sin crecer. Cierra al llegar a `fb_max_vueltas_sin_crecer`. Mantiene su tope de
    tiempo como red de seguridad.
    """
    max_vueltas_sin_crecer = fb_cfg.get("fb_max_vueltas_sin_crecer", 5)
    max_rounds = fb_cfg.get("max_scroll_rounds", 60)
    timeout_total_s = fb_cfg.get("scroll_timeout_total_s", 300)
    sin_crecer = 0
    total_clicks = 0
    t_inicio = datetime.now()

    for ronda in range(max_rounds):
        n_antes = len(acumulado)

        # Clickear un boton "N respuestas" visible por vuelta
        botones = container.locator('div[role="button"]:has-text("respuesta")').all()
        for boton in botones:
            try:
                if not boton.is_visible(timeout=500):
                    continue
                texto = boton.inner_text(timeout=1000).strip()
                if re.search(r"\d+\s*respuesta", texto, re.IGNORECASE):
                    boton.scroll_into_view_if_needed(timeout=1000)
                    page.wait_for_timeout(200)
                    boton.click(timeout=1000, force=True)
                    page.wait_for_timeout(1500)
                    total_clicks += 1
                    break
            except Exception:
                pass

        _recolectar_visibles(container, acumulado)
        crecio = len(acumulado) - n_antes

        if crecio > 0:
            sin_crecer = 0
            logger.debug("[respuestas] ronda %d: +%d nuevas (total %d)",
                         ronda + 1, crecio, len(acumulado))
            if output_path and operador:
                _guardar_acumulado_incremental(acumulado, output_path, operador, page.url)
        else:
            sin_crecer += 1
            logger.debug("[respuestas] ronda %d: sin crecer (%d/%d)",
                         ronda + 1, sin_crecer, max_vueltas_sin_crecer)

        # PARADA PRINCIPAL: dejo de crecer
        if sin_crecer >= max_vueltas_sin_crecer:
            logger.debug("[respuestas]: cerrado por %d vueltas sin crecer", max_vueltas_sin_crecer)
            break
        # red de seguridad
        if int((datetime.now() - t_inicio).total_seconds()) >= timeout_total_s:
            logger.debug("[respuestas]: parado por timeout total (%ds) — posible incompleto", timeout_total_s)
            break

    logger.info("Respuestas expandidas: %d clicks", total_clicks)


# ---------------------------------------------------------------------------
# Extraccion de comentarios del DOM
# ---------------------------------------------------------------------------

_SELECTOR_COMENTARIOS = (
    '[aria-label^="Comentario de"], [aria-label^="Respuesta de"], '
    '[aria-label^="Comment by"], [aria-label^="Reply by"]'
)


def _parse_comment_element(elem) -> Optional[dict]:
    """
    Extrae los datos de un unico elemento de comentario/respuesta del DOM.
    Devuelve un dict con tipo/nombre/texto/etc o None si no se pudo parsear.
    """
    try:
        aria = elem.get_attribute("aria-label", timeout=0) or ""
        nombre, tiempo, tipo, respondiendo_a = _parse_aria_label(aria)
        if tipo is None:
            return None

        texto_div = elem.locator('div[style="text-align: start;"]').first
        texto = texto_div.inner_text(timeout=0) if texto_div.count() > 0 else elem.inner_text(timeout=0)

        reacciones = 0
        for div in elem.locator("div[aria-label]").all():
            label = div.get_attribute("aria-label", timeout=0) or ""
            m_r = re.search(r"(\d+)\s*reacci", label, re.IGNORECASE)
            if m_r:
                reacciones = int(m_r.group(1))
                break

        return {
            "tipo": tipo,
            "nombre_usuario": nombre,
            "tiempo": tiempo,
            "texto": texto,
            "reacciones": reacciones,
            "respondiendo_a": respondiendo_a,
            "aria_label": aria,
        }
    except Exception as e:
        logger.debug("Error parseando elemento de comentario: %s", e)
        return None


def _recolectar_visibles(container, acumulado: dict):
    """
    Recorre los comentarios actualmente visibles en el DOM y los agrega a
    'acumulado' (dict keyed por aria_label, preserva orden de insercion).
    Llamar repetidamente durante el scroll: cada elemento se guarda la primera
    vez que aparece, resistiendo la virtualizacion del DOM de Facebook.
    """
    nuevos = 0
    for elem in container.locator(_SELECTOR_COMENTARIOS).all():
        data = _parse_comment_element(elem)
        if not data:
            continue
        clave = data["aria_label"]
        if clave and clave not in acumulado:
            acumulado[clave] = data
            nuevos += 1
    return nuevos


def _construir_jerarquia(acumulado: dict) -> list:
    """
    A partir del dict acumulado (en orden de aparicion), arma la lista de
    comentarios principales con sus respuestas anidadas. Cada respuesta cuelga
    del ultimo comentario principal visto antes que ella.
    """
    result = []
    ultimo_comentario = None

    for data in acumulado.values():
        if data["tipo"] == "comentario":
            comment = {
                "nombre_usuario": data["nombre_usuario"],
                "tiempo": data["tiempo"],
                "texto": data["texto"],
                "reacciones": data["reacciones"],
                "respuestas": [],
                "aria_label": data["aria_label"],
            }
            result.append(comment)
            ultimo_comentario = comment
        elif data["tipo"] == "respuesta" and ultimo_comentario is not None:
            reply = {
                "nombre_usuario": data["nombre_usuario"],
                "tiempo": data["tiempo"],
                "texto": data["texto"],
                "reacciones": data["reacciones"],
                "aria_label": data["aria_label"],
            }
            if data["respondiendo_a"]:
                reply["respondiendo_a"] = data["respondiendo_a"]
            ultimo_comentario["respuestas"].append(reply)

    return result


def _parse_aria_label(aria: str) -> tuple:
    nombre, tiempo, tipo, respondiendo_a = None, None, None, None

    if aria.startswith("Comentario de "):
        tipo = "comentario"
        resto = aria[14:]
        m = re.match(r"(.+?)\s+(Hace .+|\d+\s*[a-z]+)$", resto)
        if m:
            nombre, tiempo = m.group(1).strip(), m.group(2).strip()
        else:
            nombre = resto

    elif aria.startswith("Respuesta de "):
        tipo = "respuesta"
        resto = aria[13:]
        m = re.match(r"(.+?)\s+al comentario de\s+(.+?)\s+(Hace .+|\d+\s*[a-z]+)$", resto)
        if m:
            nombre, respondiendo_a, tiempo = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        else:
            nombre = resto

    elif aria.startswith("Comment by "):
        tipo = "comentario"
        resto = aria[11:]
        m = re.match(r"(.+?)\s+(\d+[a-z]+ ago|Just now)$", resto)
        if m:
            nombre, tiempo = m.group(1).strip(), m.group(2).strip()
        else:
            nombre = resto

    elif aria.startswith("Reply by "):
        tipo = "respuesta"
        resto = aria[9:]
        m = re.match(r"(.+?)\s+to\s+(.+?)'s comment\s+(\d+[a-z]+ ago|Just now)$", resto)
        if m:
            nombre, respondiendo_a, tiempo = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        else:
            nombre = resto

    return nombre, tiempo, tipo, respondiendo_a


# ---------------------------------------------------------------------------
# Extraccion GraphQL
# ---------------------------------------------------------------------------

def _extract_nodes_from_graphql(data, nodes=None) -> list:
    if nodes is None:
        nodes = []
    if isinstance(data, dict):
        if "edges" in data and isinstance(data["edges"], list):
            for edge in data["edges"]:
                if "node" in edge:
                    node = edge["node"]
                    if "id" in node and ("body" in node or "text" in node or "message" in node):
                        nodes.append(node)
        for value in data.values():
            _extract_nodes_from_graphql(value, nodes)
    elif isinstance(data, list):
        for item in data:
            _extract_nodes_from_graphql(item, nodes)
    return nodes


def _process_graphql_nodes(raw_nodes: list) -> list:
    todos = {}
    principales = []
    respuestas = []

    for node in raw_nodes:
        data = _extract_comment_data_from_node(node)
        if not data:
            continue
        todos[data["id"]] = {"data": data, "raw": node}
        if data["depth"] == 0:
            principales.append(data)
        else:
            respuestas.append(data)

    anidadas = 0
    for resp in respuestas:
        raw = todos.get(resp["id"], {}).get("raw", {})
        parent_id = raw.get("feedback", {}).get("parent_object_ent", {}).get("id")
        if parent_id:
            padre = next((c for c in principales if c["id"] == parent_id), None)
            if not padre and parent_id in todos:
                padre = todos[parent_id]["data"].copy()
                padre["respuestas"] = []
                principales.append(padre)
            if padre and resp["id"] != padre["id"]:
                padre["respuestas"].append(resp)
                anidadas += 1

    logger.debug("GraphQL: %d principales, %d/%d respuestas anidadas", len(principales), anidadas, len(respuestas))
    return principales


def _extract_comment_data_from_node(node: dict) -> Optional[dict]:
    try:
        ts = node.get("created_time", 0)
        feedback = node.get("feedback", {})
        reactors = feedback.get("unified_reactors") or feedback.get("reactors", {})
        reacciones = reactors.get("count", 0) if isinstance(reactors, dict) else 0
        return {
            "id": node.get("id", ""),
            "tipo": "comentario" if node.get("depth", 0) == 0 else "respuesta",
            "depth": node.get("depth", 0),
            "texto": node.get("body", {}).get("text", ""),
            "autor": node.get("author", {}).get("name", ""),
            "autor_id": node.get("author", {}).get("id", ""),
            "timestamp": ts,
            "fecha": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            "reacciones": reacciones,
            "respuestas": [],
        }
    except Exception as e:
        logger.debug("Error extrayendo nodo GraphQL: %s", e)
        return None


def _extract_count(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else "0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Facebook: extrae comentarios de posts listados")
    parser.add_argument("--periodo", required=True, metavar="YYYYMM", help="Periodo a procesar, ej: 202506")
    parser.add_argument("--operators", nargs="+", metavar="NOMBRE", help="Filtrar operadores (opcional), ej: entel claro")
    args = parser.parse_args()
    scrape_all_operators(periodo=args.periodo, operators=args.operators)
