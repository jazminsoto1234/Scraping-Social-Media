"""
Paso 1 del pipeline Facebook: navega el perfil de cada operadora, scrollea el feed,
extrae URLs de posts/reels desde los hrefs de los comentarios del feed, deduce la fecha
del texto relativo visible ("1 día", "6 días", "15 de mayo") y para de scrollear en
cuanto los posts son más antiguos que el periodo solicitado.

Por qué este enfoque (confirmado por diagnóstico DOM):
  - Facebook renderiza [role="article"] = comentarios destacados de cada post.
    Cada comentario lleva un href al post padre (con comment_id).
  - Limpiar ese href da la URL canónica del post.
  - La fecha está ofuscada en spans individuales por letra — no hay data-utime ni title.
    Se deduce del texto relativo del <a> que apunta al post:
      "1 día"        → hoy - 1 día
      "6 días"       → hoy - 6 días
      "1 semana"     → hoy - 7 días
      "15 de mayo"   → 2026-05-15 (año actual, o anterior si sería futuro)
      "hace 2 horas" → hoy

Uso:
    python src/scrapers/facebook/profile_post_lister.py --periodo 202506
    python src/scrapers/facebook/profile_post_lister.py --periodo 202506 --operators entel claro
"""
import re
import json
import logging
import argparse
import calendar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from playwright.sync_api import sync_playwright, Page

logging.basicConfig(
    filename="facebook_post_lister.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _cargar_config() -> dict:
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Utilidades de URL
# ---------------------------------------------------------------------------

def _extraer_post_id(url: str) -> Optional[str]:
    m = re.search(r"(pfbid[\w]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/reel/(\d+)", url)
    if m:
        return f"reel_{m.group(1)}"
    m = re.search(r"/posts/(\d+)", url)
    if m:
        return f"post_{m.group(1)}"
    return None


def _es_url_post(url: str) -> bool:
    return "/posts/" in url or "pfbid" in url or "/reel/" in url


def _limpiar_url(href: str) -> Optional[str]:
    if not href or not _es_url_post(href):
        return None
    url = href.split("?")[0].rstrip("/")
    if not url.startswith("http"):
        url = f"https://www.facebook.com{url}"
    return url


# ---------------------------------------------------------------------------
# Deducir fecha desde texto relativo del feed
# ---------------------------------------------------------------------------

_MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_MESES_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _fecha_desde_texto_relativo(texto: str, hoy: datetime) -> Optional[datetime]:
    """
    Convierte el texto de tiempo relativo de Facebook a datetime.
    Ejemplos:
      "1 día"        → hoy - 1d
      "6 días"       → hoy - 6d
      "1 semana"     → hoy - 7d
      "2 semanas"    → hoy - 14d
      "1 mes"        → hoy - 30d
      "hace 3 horas" → hoy
      "15 de mayo"   → año_actual-05-15 (o año anterior si sería futuro)
      "3 de junio"   → año_actual-06-03
    """
    t = texto.strip().lower()

    # "hace N horas/minutos" → hoy
    if re.search(r"hace\s+\d+\s*(hora|minuto|segundo|h\b|min)", t):
        return hoy.replace(hour=0, minute=0, second=0, microsecond=0)

    # "N día(s)"
    m = re.match(r"(\d+)\s+d[ií]a", t)
    if m:
        return (hoy - timedelta(days=int(m.group(1)))).replace(
            hour=0, minute=0, second=0, microsecond=0)

    # "N semana(s)" — Facebook abrevia: "sem", "s", "w", "wk"
    m = re.match(r"(\d+)\s*(semana|sem\b|wk?s?\b)", t)
    if m:
        return (hoy - timedelta(weeks=int(m.group(1)))).replace(
            hour=0, minute=0, second=0, microsecond=0)

    # "N mes(es)" — abreviatura "m" o "mo"
    m = re.match(r"(\d+)\s*(mes|mo\b)", t)
    if m:
        return (hoy - timedelta(days=int(m.group(1)) * 30)).replace(
            hour=0, minute=0, second=0, microsecond=0)

    # "N hora(s)" sin "hace" — abrev "h", "hr"
    m = re.match(r"(\d+)\s*(hora|hr?\b)", t)
    if m:
        return hoy.replace(hour=0, minute=0, second=0, microsecond=0)

    # "N min(utos)" sin "hace"
    m = re.match(r"(\d+)\s*(min|minuto)", t)
    if m:
        return hoy.replace(hour=0, minute=0, second=0, microsecond=0)

    # "15 de mayo", "3 de junio" (sin año — Facebook omite el año si es el actual)
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)", t)
    if m:
        dia = int(m.group(1))
        mes = _MESES_ES.get(m.group(2)) or _MESES_EN.get(m.group(2))
        if mes:
            anio = hoy.year
            try:
                fecha = datetime(anio, mes, dia)
                # Si la fecha calculada es futura, es del año anterior
                if fecha > hoy:
                    fecha = datetime(anio - 1, mes, dia)
                return fecha
            except ValueError:
                pass

    # "mayo 15", "june 3" (formato inglés sin año)
    m = re.search(r"(\w+)\s+(\d{1,2})$", t)
    if m:
        mes = _MESES_ES.get(m.group(1)) or _MESES_EN.get(m.group(1))
        if mes:
            dia = int(m.group(2))
            anio = hoy.year
            try:
                fecha = datetime(anio, mes, dia)
                if fecha > hoy:
                    fecha = datetime(anio - 1, mes, dia)
                return fecha
            except ValueError:
                pass

    return None


def _extraer_texto_fecha_articulo(page: Page, idx: int) -> Optional[str]:
    """
    Extrae el texto de tiempo relativo del <a> que apunta al post padre
    dentro del artículo (es el link cuyo href tiene /posts/ o /reel/).
    Ese <a> siempre muestra el tiempo relativo: "1 día", "6 días", "15 de mayo".
    """
    resultado = page.evaluate(f"""
        () => {{
            let arts = document.querySelectorAll('[role="article"]');
            let art = arts[{idx}];
            if (!art) return null;
            // El <a> que apunta al post tiene el tiempo relativo como texto
            let links = Array.from(art.querySelectorAll('a[href]'));
            for (let a of links) {{
                let h = a.href || '';
                if (h.includes('/posts/') || h.includes('pfbid') || h.includes('/reel/')) {{
                    let txt = (a.innerText || a.textContent || '').trim();
                    if (txt.length > 0 && txt.length < 60) return txt;
                }}
            }}
            return null;
        }}
    """)
    return resultado


# ---------------------------------------------------------------------------
# GraphQL — atajo de fecha (respaldo)
# ---------------------------------------------------------------------------

def _recorrer_graphql(data, ts_por_id: dict):
    if isinstance(data, list):
        for item in data:
            _recorrer_graphql(item, ts_por_id)
        return
    if not isinstance(data, dict):
        return
    if "creation_time" in data:
        ts = data.get("creation_time")
        for campo in ("wwwURL", "url", "permalink_url", "link_url"):
            val = data.get(campo, "")
            uri = val if isinstance(val, str) else (val.get("uri", "") if isinstance(val, dict) else "")
            if _es_url_post(uri):
                pid = _extraer_post_id(uri)
                if pid and isinstance(ts, int) and ts > 0 and pid not in ts_por_id:
                    ts_por_id[pid] = ts
                    logger.debug("GraphQL: pid=%s ts=%s", pid, ts)
                break
    for val in data.values():
        _recorrer_graphql(val, ts_por_id)


def _registrar_interceptor(page: Page, ts_por_id: dict):
    def _handle(response):
        if "/api/graphql/" not in response.url:
            return
        try:
            _recorrer_graphql(response.json(), ts_por_id)
        except Exception:
            pass
    page.on("response", _handle)


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------

def _scroll(page: Page, px: int = 1000, espera_ms: int = 3000):
    page.evaluate(f"window.scrollBy(0, {px})")
    page.wait_for_timeout(espera_ms)


def _destrabar(page: Page):
    logger.debug("Destrabando: scroll arriba→abajo")
    page.evaluate("window.scrollBy(0, -800)")
    page.wait_for_timeout(1800)
    page.evaluate("window.scrollBy(0, 1600)")
    page.wait_for_timeout(3000)


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def list_profile_posts(page: Page, profile_url: str, operador: str,
                       date_from: str, date_to: str, out_path: Path) -> list:
    dt_from = datetime.strptime(date_from, "%Y-%m-%d")
    dt_to   = datetime.strptime(date_to,   "%Y-%m-%d")
    hoy     = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    posts_encontrados: list = []
    seen_ids: set = set()
    ts_por_id: dict = {}
    fuera_de_rango = 0
    sin_links_nuevos = 0

    def _guardar():
        out_path.write_text(json.dumps({
            "operador":    operador,
            "perfil_url":  profile_url,
            "date_from":   date_from,
            "date_to":     date_to,
            "total_posts": len(posts_encontrados),
            "posts":       posts_encontrados,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    _registrar_interceptor(page, ts_por_id)

    # Navegar al perfil
    url_actual = page.url
    if profile_url.split("?")[0].rstrip("/") not in url_actual:
        print(f"  Navegando a {profile_url} ...")
        page.goto(profile_url, wait_until="domcontentloaded", timeout=40000)
        page.wait_for_timeout(5000)
    else:
        print(f"  Ya en {url_actual}")
        page.wait_for_timeout(2000)

    # Esperar primer artículo
    print("  Esperando que cargue el feed...")
    feed_listo = False
    for intento in range(20):
        _scroll(page, px=600, espera_ms=2500)
        if page.locator('[role="article"]').count() >= 1:
            print(f"  Feed listo (intento {intento+1})")
            feed_listo = True
            break
        print(f"  Esperando... intento {intento+1}/20")
        if (intento + 1) % 4 == 0:
            _destrabar(page)

    if not feed_listo:
        print("  ERROR: el feed no cargó. Verifica sesión activa.")
        logger.warning("Feed no cargó en %s", profile_url)
        return []

    # Loop: scroll → links nuevos → fecha del texto relativo → filtro → ¿parar?
    # Rastreamos por href (estable), no por índice DOM (FB virtualiza y cambia índices)
    scroll_num = 0

    while sin_links_nuevos < 6 and fuera_de_rango < 3:
        scroll_num += 1
        _scroll(page)

        # Extraer hrefs de TODOS los artículos visibles ahora en el DOM.
        # Por cada artículo se agrupan los links por pid y se conserva el texto
        # más antiguo (mayor número de días) — el link al post tiene el tiempo
        # de publicación; los links a comentarios tienen tiempos más recientes.
        hrefs_por_idx = page.evaluate("""
            () => {
                let arts = document.querySelectorAll('[role="article"]');
                let resultado = {};
                arts.forEach((art, idx) => {
                    // Agrupar por URL base (sin query params) para quedarnos
                    // con el texto de tiempo más largo (más antiguo) de cada post
                    let mapa = {};
                    Array.from(art.querySelectorAll('a[href]'))
                        .filter(a => a.href.includes('/posts/') || a.href.includes('pfbid') || a.href.includes('/reel/'))
                        .forEach(a => {
                            let url_base = a.href.split('?')[0].replace(/\\/$/, '');
                            let txt = (a.innerText || a.textContent || '').trim().slice(0, 60);
                            // Guardar el texto más largo (más antiguo) para esta URL
                            if (!mapa[url_base] || txt.length > mapa[url_base].texto.length) {
                                mapa[url_base] = { href: a.href, texto: txt };
                            }
                        });
                    let links = Object.values(mapa);
                    if (links.length > 0) resultado[idx] = links;
                });
                return resultado;
            }
        """)

        links_nuevos_count = 0
        for idx_str, links in hrefs_por_idx.items():
            for link in links:
                href  = link["href"]
                texto = link["texto"]  # texto más antiguo del artículo para este post

                url = _limpiar_url(href)
                if not url:
                    continue
                pid = _extraer_post_id(url)
                if not pid or pid in seen_ids:
                    continue

                seen_ids.add(pid)
                links_nuevos_count += 1

                # Deducir fecha: GraphQL primero, luego texto relativo
                fecha: Optional[datetime] = None
                if pid in ts_por_id:
                    try:
                        fecha = datetime.fromtimestamp(ts_por_id[pid]).replace(
                            hour=0, minute=0, second=0, microsecond=0)
                        logger.debug("Fecha GraphQL pid=%s: %s", pid, fecha.date())
                    except Exception:
                        pass

                if fecha is None and texto:
                    fecha = _fecha_desde_texto_relativo(texto, hoy)
                    if fecha:
                        logger.debug("Fecha relativa '%s' → %s", texto, fecha.date())

                # Filtrar por periodo
                if fecha:
                    if fecha > dt_to:
                        logger.debug("pid=%s fecha=%s > dt_to, saltando", pid, fecha.date())
                        fuera_de_rango = 0
                        continue
                    if fecha < dt_from:
                        fuera_de_rango += 1
                        print(f"  {url[:60]} → {fecha.date()} anterior al periodo ({fuera_de_rango}/3)")
                        if fuera_de_rango >= 3:
                            print("  3 posts fuera del periodo — parando.")
                            return posts_encontrados
                        continue
                    fuera_de_rango = 0
                else:
                    logger.warning("Sin fecha para pid=%s texto=%r — guardando igual", pid, texto)

                entry = {
                    "url":   url,
                    "fecha": fecha.strftime("%Y-%m-%d") if fecha else None,
                }
                posts_encontrados.append(entry)
                print(f"  + {url[:70]}  [{entry['fecha'] or 'sin fecha'}]  ({texto!r})")
                _guardar()

        if links_nuevos_count == 0:
            sin_links_nuevos += 1
            print(f"  Scroll {scroll_num}: sin posts nuevos ({sin_links_nuevos}/6)")
            if sin_links_nuevos % 2 == 0:
                _destrabar(page)
        else:
            sin_links_nuevos = 0
            print(f"  Scroll {scroll_num}: +{links_nuevos_count} nuevos (total guardados={len(posts_encontrados)})")

    return posts_encontrados


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paso 1 Facebook: listar URLs de posts por periodo")
    parser.add_argument("--periodo",   required=True, metavar="YYYYMM", help="Periodo, ej: 202506")
    parser.add_argument("--operators", nargs="+",     metavar="NOMBRE", help="Filtrar operadores (opcional)")
    args = parser.parse_args()

    config = _cargar_config()
    todos = config.get("operators_facebook", {})
    if args.operators:
        todos = {k: v for k, v in todos.items() if k in args.operators}

    if not todos:
        print("ERROR: No hay operadores en 'operators_facebook' en config.yaml")
        return

    year  = int(args.periodo[:4])
    month = int(args.periodo[4:])
    date_from = f"{year}-{month:02d}-01"
    last_day  = calendar.monthrange(year, month)[1]
    date_to   = f"{year}-{month:02d}-{last_day}"

    listas_dir = Path("outputs") / "listas" / args.periodo
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
            existing_pages = context.pages
            if not existing_pages:
                print("ERROR: No hay pestañas abiertas. Abre Facebook manualmente primero.")
                return
            page = next((pg for pg in existing_pages if "facebook.com" in pg.url), existing_pages[0])
            print(f"  Reusando pestaña: {page.url[:60]}")

            for operador, url_perfil in todos.items():
                if not url_perfil:
                    print(f"\nSKIP {operador}: URL no configurada")
                    continue

                print(f"\n{'='*60}")
                print(f"Operador: {operador}  |  {date_from} → {date_to}")
                print(f"{'='*60}")

                out_path = listas_dir / f"{operador}_posts.json"
                posts = list_profile_posts(
                    page, url_perfil, operador,
                    date_from, date_to, out_path
                )

                print(f"\n  Total: {out_path.name}  ({len(posts)} posts)")
                for i, p in enumerate(posts, 1):
                    print(f"  {i}. [{p.get('fecha') or 'sin fecha'}] {p['url'][:80]}")

        finally:
            browser.close()


if __name__ == "__main__":
    main()
