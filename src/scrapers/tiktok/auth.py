"""
Gestión de sesión para TikTok.
Primera ejecución: resuelve CAPTCHA manualmente y guarda sesión.
Ejecuciones posteriores: carga sesión guardada sin CAPTCHA.
"""
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


def save_session(username: str, session_file: str, config: dict):
    """Abre TikTok, espera que el usuario resuelva el CAPTCHA y guarda la sesión."""
    session_path = Path(session_file)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    tt_cfg = config["tiktok"]
    url = tt_cfg["profile_url"].format(username=username)

    print("=" * 60)
    print("GUARDANDO SESION TIKTOK")
    print("=" * 60)
    print(f"1. Se abrira el perfil: {url}")
    print("2. Si aparece CAPTCHA, resulevelo manualmente")
    print("3. Espera 20 segundos y la sesion se guardara automaticamente")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport=tt_cfg["viewport"],
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="es-US",
            timezone_id="America/Lima",
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        page = context.new_page()
        page.goto(url)
        print("\nEsperando 20 segundos para resolver CAPTCHA si aparece...")
        page.wait_for_timeout(20000)
        page.evaluate("window.scrollBy(0, 1000)")
        page.wait_for_timeout(3000)

        context.storage_state(path=str(session_path))
        print(f"\nSesion guardada en: {session_path}")
        browser.close()


def check_session(session_file: str) -> bool:
    """Verifica si existe un archivo de sesion valido."""
    path = Path(session_file)
    if not path.exists():
        logger.error("No existe sesion guardada: %s", session_file)
        print(f"\nNo existe sesion. Ejecuta primero:")
        print(f"  python main.py tiktok auth --username <usuario>")
        return False
    return True
