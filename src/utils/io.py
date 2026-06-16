import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def save_json(data, output_dir: Path, filename: str) -> Path:
    """Guarda datos como JSON en output_dir/filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Guardado: %s (%d items)", filepath, len(data) if isinstance(data, list) else 1)
    return filepath


def timestamped_filename(base: str, ext: str = "json") -> str:
    """Genera nombre de archivo con timestamp para evitar sobreescrituras."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}.{ext}"


def setup_logging(log_file: str = None, level: int = logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
