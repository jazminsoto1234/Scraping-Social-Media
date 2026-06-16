import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_ROOT / "configs" / ".env")


def load_config(config_path: str = None) -> dict:
    """Carga config.yaml desde configs/ o la ruta indicada."""
    path = Path(config_path) if config_path else _ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_session_path(platform: str, config: dict) -> Path:
    """Devuelve la ruta absoluta del archivo de sesión."""
    key = f"{platform}_session_file" if platform == "tiktok" else None
    if platform == "tiktok":
        rel = config["tiktok"]["session_file"]
    else:
        rel = config["facebook"]["session_file"]
    return _ROOT / rel


def get_output_dir(platform: str, config: dict) -> Path:
    """Devuelve y crea el directorio de salida para la plataforma."""
    key = f"{platform}_dir"
    rel = config["outputs"][key]
    path = _ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    return path
