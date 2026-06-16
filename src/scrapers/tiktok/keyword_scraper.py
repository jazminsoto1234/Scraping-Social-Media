"""
Scraper de TikTok por keyword.
Estrategia: busca en /search/video, captura los items via API,
luego abre cada video con CLICK (igual que el flujo de usuario)
para forzar el layout modal donde api/comment/list se dispara via XHR.
"""
import re
import logging
from datetime import datetime
from typing import Optional

from playwright.sync_api import sync_playwright, Page

from .scraper import (
    _build_browser,
    _wait_for_captcha,
    _simplify_video_meta,
    _simplify_comments,
    _filter_by_date,
    _sort_by_engagement,
    _expand_replies_visible,
    _expand_replies_final,
)

logger = logging.getLogger(__name__)


def search_by_keyword(
    keyword: str,
    top_n: int,
    session_file: str,
    config: dict,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    exclude_users: Optional[list[str]] = None,
    max_replies: Optional[int] = None,
) -> list[dict]:
    """
    Busca 'keyword' en TikTok y extrae los top_n videos con sus comentarios.

    Estrategia:
      1. Navega a /search/video y captura los items via api/search.
      2. Hace scroll en la página de resultados para cargar las miniaturas.
      3. Abre cada video con CLICK (abre el modal viewer) para que TikTok
         dispare api/comment/list via XHR — igual que search_by_user.

    Args:
        keyword: termino de busqueda.
        top_n: cantidad de videos a procesar.
        session_file: ruta al storage_state.json.
        config: diccionario con la configuracion del proyecto.
        date_from: filtro opcional de fecha inicio (YYYY-MM-DD).
        date_to: filtro opcional de fecha fin (YYYY-MM-DD).
        exclude_users: unique_id de usuarios cuyos comentarios/replies se excluyen.
        max_replies: maximo de replies a conservar por comentario padre (None = sin limite).

    Returns:
        Lista de dicts con {video, comments}.
    """
    tt_cfg = config["tiktok"]
    search_url = f"https://www.tiktok.com/search/video?q={keyword.replace(' ', '+')}"
    results = []

    with sync_playwright() as p:
        browser, context, page = _build_browser(p, session_file, tt_cfg)
        captured_videos = []

        def handle_search(response):
            if "api/search/item/full" in response.url or "api/search/general/full" in response.url:
                try:
                    data = response.json()
                    items = data.get("item_list", data.get("itemList", []))
                    if items:
                        captured_videos.extend(items)
                        logger.info(
                            "Videos de busqueda capturados: +%d (total %d)",
                            len(items), len(captured_videos),
                        )
                except Exception:
                    pass

        page.on("response", handle_search)
        logger.info("Buscando keyword: %s", keyword)
        page.goto(search_url)
        page.wait_for_timeout(5000)

        # Scroll para cargar mas resultados de busqueda
        for _ in range(3):
            page.evaluate("window.scrollBy(0, 1500)")
            page.wait_for_timeout(2000)

        page.remove_listener("response", handle_search)

        items = _filter_by_date(captured_videos, date_from, date_to)
        items = _sort_by_engagement(items)[:top_n]
        logger.info("Videos filtrados y ordenados: %d", len(items))

        if not items:
            browser.close()
            return results

        # Localizar las miniaturas visibles en la pagina de busqueda
        # TikTok usa data-e2e="search_video-item" o "search-card-item" en los resultados
        video_elements = _get_search_video_elements(page)
        logger.info("Miniaturas visibles en busqueda: %d", len(video_elements))

        for idx in range(min(top_n, len(items), len(video_elements))):
            item = items[idx]
            video_meta = _simplify_video_meta(item)
            logger.info(
                "Procesando video %d/%d: %s",
                idx + 1, min(top_n, len(items), len(video_elements)),
                item.get("id", ""),
            )

            comments, replies_dict = _extract_comments_via_click(
                page, video_elements, idx, tt_cfg
            )
            logger.info(
                "Comentarios capturados para %s: %d comentarios, %d threads con replies",
                item.get("id", ""), len(comments), len(replies_dict),
            )

            comments_simplified = _simplify_comments(
                comments, replies_dict,
                exclude_users=exclude_users,
                max_replies=max_replies,
            )
            results.append({
                "video": video_meta,
                "comments": comments_simplified,
            })

            # Volver a la pagina de busqueda para el siguiente video
            if idx < min(top_n, len(items), len(video_elements)) - 1:
                page.go_back()
                page.wait_for_timeout(3000)
                video_elements = _get_search_video_elements(page)

        browser.close()

    return results


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_search_video_elements(page: Page) -> list:
    """Retorna los elementos de video visibles en la pagina de resultados de busqueda."""
    selectors = [
        '[data-e2e="search_video-item"]',
        '[data-e2e="search-card-item"]',
        '[class*="DivItemContainer"]',
    ]
    for sel in selectors:
        try:
            elements = page.locator(sel).all()
            if elements:
                logger.debug("Selector de miniaturas: %s (%d elementos)", sel, len(elements))
                return elements
        except Exception:
            pass
    logger.warning("No se encontraron miniaturas con selectores conocidos")
    return []


def _extract_comments_via_click(
    page: Page, video_elements: list, idx: int, tt_cfg: dict
) -> tuple[list, dict]:
    """
    Hace click en una miniatura de video y extrae comentarios via interceptacion XHR.
    Replica el flujo simple de test_primer_video.py.
    """
    all_comments = []
    all_replies = {}

    def handle_response(response):
        url = response.url
        if "api/comment/list" in url and "reply" not in url:
            try:
                data = response.json()
                comments = data.get("comments", [])
                if comments:
                    all_comments.extend(comments)
                    logger.debug("comment/list: +%d comentarios (total %d)", len(comments), len(all_comments))
            except Exception as e:
                logger.debug("Error parseando comment/list: %s", e)

        if "/api/comment/list/reply/" in url:
            try:
                data = response.json()
                replies = data.get("comments", [])
                if replies:
                    match = re.search(r"comment_id=(\d+)", url)
                    if match:
                        pid = match.group(1)
                        all_replies.setdefault(pid, []).extend(replies)
                        logger.debug("reply: +%d respuestas para comentario %s", len(replies), pid)
            except Exception as e:
                logger.debug("Error parseando reply: %s", e)

    page.on("response", handle_response)
    try:
        logger.info("Clickeando video %d", idx)
        video_elements[idx].click()
        page.wait_for_timeout(3000)

        logger.info("Scrolleando comentarios")
        prev_count = 0
        sin_cambios = 0
        scroll_iter = 0

        while True:
            scroll_iter += 1
            page.evaluate("""
                const container = document.querySelector('[class*="DivCommentListContainer"]') ||
                                 document.querySelector('[data-e2e="browse-comment"]');
                if (container) container.scrollTop = container.scrollHeight;
            """)
            page.wait_for_timeout(100)

            _expand_replies_visible(page, all_replies, None)

            current_count = len(all_comments)
            if current_count > prev_count:
                sin_cambios = 0
            else:
                sin_cambios += 1

            if sin_cambios >= 5:
                break

            prev_count = current_count

        logger.info("Expandiendo replies finales")
        _expand_replies_final(page, all_replies)

    except Exception as e:
        logger.error("Error al procesar video %d: %s", idx, e)

    page.remove_listener("response", handle_response)
    return all_comments, all_replies
