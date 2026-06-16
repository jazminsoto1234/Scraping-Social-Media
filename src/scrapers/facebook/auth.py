"""
Gestion de sesion para Facebook.
Primera ejecucion: login manual y se guarda la sesion.
Ejecuciones posteriores: carga la sesion guardada.
"""
import json
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


def save_session(session_file: str, config: dict):
    """Abre Facebook, espera login manual del usuario y guarda la sesion."""
    session_path = Path(session_file)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    fb_cfg = config["facebook"]

    print("=" * 60)
    print("GUARDANDO SESION FACEBOOK")
    print("=" * 60)
    print("1. Se abrira Facebook en el navegador")
    print("2. INICIA SESION MANUALMENTE con tu cuenta")
    print("3. Navega hasta estar completamente logueado")
    print("4. Presiona ENTER aqui cuando hayas terminado")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport=fb_cfg["viewport"],
            user_agent=fb_cfg["user_agent"],
            locale=fb_cfg["locale"],
            timezone_id=fb_cfg["timezone"],
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        page = context.new_page()
        print("\nAbriendo Facebook...")
        page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        input("\n>>> Presiona ENTER despues de haber iniciado sesion completamente...")

        # Verificar sesion activa
        logged_in = (
            page.locator('[aria-label="Cuenta"]').count() > 0
            or page.locator('[aria-label="Menu de la cuenta"]').count() > 0
        )
        if not logged_in:
            print("Advertencia: no se detectaron elementos de sesion activa.")
            resp = input("Continuar de todas formas? (s/n): ")
            if resp.lower() != "s":
                print("Cancelado.")
                browser.close()
                return

        storage_state = context.storage_state()
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(storage_state, f, indent=2)

        print(f"\nSesion guardada en: {session_path}")
        print(f"Cookies guardadas: {len(storage_state.get('cookies', []))}")
        browser.close()


def check_session(session_file: str) -> bool:
    """Verifica si existe un archivo de sesion valido."""
    path = Path(session_file)
    if not path.exists():
        logger.error("No existe sesion guardada: %s", session_file)
        print(f"\nNo existe sesion. Ejecuta primero:")
        print(f"  python main.py facebook auth")
        return False
    return True
