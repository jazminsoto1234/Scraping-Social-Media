"""
Extraccion de transcripciones de videos.
Metodo 1: subtitulos embebidos de la plataforma.
Metodo 2: descarga de audio + Whisper (ASR).
"""
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_transcript(video: dict, config: dict, output_dir: Optional[Path] = None) -> dict:
    """
    Obtiene la transcripcion de un video.
    Intenta primero subtitulos de plataforma; si no hay, usa Whisper.

    Args:
        video: dict con metadata del video (debe tener 'id' y opcionalmente 'subtitles').
        config: configuracion del proyecto.
        output_dir: directorio donde guardar el audio temporal (si se usa Whisper).

    Returns:
        Dict con {video_id, transcript, language, method}.
    """
    video_id = video.get("id", "unknown")

    # Intentar subtitulos embebidos (TikTok los incluye en la respuesta de la API)
    subtitle_text = _extract_platform_subtitles(video)
    if subtitle_text:
        logger.info("Transcripcion via subtitulos de plataforma: video %s", video_id)
        return {
            "video_id": video_id,
            "transcript": subtitle_text,
            "language": config.get("transcription", {}).get("language", "es"),
            "method": "platform_subtitles",
        }

    # Fallback: descargar audio y usar Whisper
    video_url = _get_video_url(video)
    if not video_url:
        logger.warning("No hay URL de video disponible para: %s", video_id)
        return {
            "video_id": video_id,
            "transcript": "",
            "language": "",
            "method": "none",
        }

    return _transcribe_with_whisper(video_id, video_url, config, output_dir)


def _extract_platform_subtitles(video: dict) -> str:
    """Extrae subtitulos embebidos de la respuesta de la API de TikTok."""
    # TikTok incluye subtitulos en video.subtitleInfos o video.subtitles
    subtitle_infos = (
        video.get("video", {}).get("subtitleInfos")
        or video.get("subtitleInfos")
        or []
    )
    if not subtitle_infos:
        return ""

    # Preferir espanol, luego cualquier otro
    preferred = None
    for sub in subtitle_infos:
        lang = sub.get("LanguageCodeName", "").lower()
        if "es" in lang or "spa" in lang:
            preferred = sub
            break
    if not preferred and subtitle_infos:
        preferred = subtitle_infos[0]

    if not preferred:
        return ""

    # El campo puede ser texto directo o una URL que requiere descarga
    if "Url" in preferred:
        return _download_subtitle(preferred["Url"])

    return preferred.get("text", "")


def _download_subtitle(url: str) -> str:
    """Descarga el contenido de un archivo de subtitulos desde URL."""
    try:
        import requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        content = resp.text
        # Parsear formato SRT simple: extraer solo las lineas de texto
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.isdigit() and "-->" not in line:
                lines.append(line)
        return " ".join(lines)
    except Exception as e:
        logger.warning("Error descargando subtitulos: %s", e)
        return ""


def _get_video_url(video: dict) -> str:
    """Extrae la URL de reproduccion del video desde la metadata."""
    # TikTok anida la URL en video.playAddr o video.downloadAddr
    return (
        video.get("video", {}).get("playAddr")
        or video.get("video", {}).get("downloadAddr")
        or video.get("playAddr")
        or ""
    )


def _transcribe_with_whisper(
    video_id: str, video_url: str, config: dict, output_dir: Optional[Path]
) -> dict:
    """Descarga el audio del video y lo transcribe con Whisper."""
    cfg = config.get("transcription", {})
    model_name = cfg.get("whisper_model", "base")
    language = cfg.get("language", "es")
    device = cfg.get("device", "cpu")
    keep_audio = cfg.get("keep_audio", False)

    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper no instalado. Ejecuta:\n"
            "  pip install openai-whisper"
        )

    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp no instalado. Ejecuta:\n"
            "  pip install yt-dlp"
        )

    # Decidir directorio de audio: output_dir si keep_audio, sino temp
    audio_dir: Optional[Path] = None
    _tmpdir_ctx = None

    if keep_audio and output_dir:
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        import contextlib
        _tmpdir_ctx = tempfile.TemporaryDirectory()
        audio_dir = Path(_tmpdir_ctx.name)

    with contextlib.ExitStack() as stack:
        if _tmpdir_ctx is not None:
            stack.enter_context(_tmpdir_ctx)

        audio_path = audio_dir / f"{video_id}.mp3"

        logger.info(
            "Descargando audio: %s (modelo=%s, idioma=%s, device=%s, keep_audio=%s)",
            video_id, model_name, language, device, keep_audio,
        )

        download_errors: list[str] = []

        class _ErrorLogger:
            def error(self, msg, *args, **_kw):
                download_errors.append(msg % args if args else msg)
            def warning(self, msg, *args, **_kw):
                logger.warning("yt-dlp: " + msg, *args)
            def debug(self, _msg, *_args, **_kw):
                pass

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_path.with_suffix("")),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "logger": _ErrorLogger(),
            "socket_timeout": 30,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            logger.error("yt-dlp lanzó excepción para %s: %s", video_id, e)
            return {"video_id": video_id, "transcript": "", "language": "", "method": "whisper_failed"}

        if download_errors:
            logger.error("Errores de descarga para %s: %s", video_id, "; ".join(download_errors))

        if not audio_path.exists():
            logger.error(
                "Archivo de audio no encontrado tras descarga: %s (¿descarga parcial o URL inválida?)",
                audio_path,
            )
            return {"video_id": video_id, "transcript": "", "language": "", "method": "whisper_failed"}

        audio_size = audio_path.stat().st_size
        logger.info("Audio descargado: %s (%.1f KB)", audio_path.name, audio_size / 1024)

        # Duración real del audio via mutagen (opcional) o ffprobe
        audio_duration_s = _get_audio_duration(audio_path)
        if audio_duration_s is not None:
            logger.info("Duración del audio: %.1f s (%.1f min)", audio_duration_s, audio_duration_s / 60)
        else:
            logger.warning("No se pudo determinar la duración del audio: %s", video_id)

        logger.info("Transcribiendo con Whisper (modelo=%s, device=%s): %s", model_name, device, video_id)
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(str(audio_path), language=language)

        segments = result.get("segments") or []
        if segments:
            transcribed_end = segments[-1].get("end", 0.0)
            logger.info(
                "Transcripción completada: %.1f s transcritos de %.1f s totales%s",
                transcribed_end,
                audio_duration_s if audio_duration_s else 0.0,
                " (POSIBLE CORTE)" if audio_duration_s and transcribed_end < audio_duration_s * 0.9 else "",
            )
        else:
            logger.warning("Whisper no produjo segmentos para: %s", video_id)

        if keep_audio and output_dir:
            logger.info("Audio conservado en: %s", audio_path)

        return {
            "video_id": video_id,
            "transcript": result.get("text", "").strip(),
            "language": result.get("language", language),
            "method": "whisper",
            "audio_duration_s": audio_duration_s,
            "transcribed_duration_s": segments[-1].get("end") if segments else None,
        }


def _get_audio_duration(audio_path: Path) -> Optional[float]:
    """Retorna la duración en segundos del archivo de audio, o None si no se puede determinar."""
    # Intentar con mutagen (liviano, sin subprocess)
    try:
        from mutagen.mp3 import MP3
        audio = MP3(str(audio_path))
        return audio.info.length
    except Exception:
        pass

    # Fallback: ffprobe
    try:
        import subprocess
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


# Stopwords en español/inglés para descartar palabras vacías al extraer keywords.
_STOPWORDS_TITULO = {
    "de", "la", "el", "en", "y", "a", "que", "los", "del", "se",
    "las", "un", "por", "con", "no", "una", "su", "para", "es",
    "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o",
    "este", "sí", "porque", "esta", "entre", "cuando", "muy",
    "sin", "sobre", "también", "me", "hasta", "hay", "donde",
    "han", "quien", "están", "estado", "desde", "todo", "nos",
    "durante", "todos", "uno", "les", "ni", "contra", "otros",
    "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes",
    "algunos", "qué", "unos", "yo", "otro", "otras", "él",
    "tanto", "esa", "estos", "mucho", "quienes", "nada",
    "muchos", "cual", "poco", "ella", "estar", "estas",
    "algunas", "algo", "nosotros", "si", "ha", "fue", "ser",
    "tiene", "tengo", "puede", "solo", "bien", "así", "ahora",
    "aquí", "the", "is", "in", "it", "of", "and", "to", "ahí",
    "esperas", "demás", "revisa", "condiciones", "uso",
}


def extraer_keyword_publicacion(descripcion: str, max_palabras: int = 3, max_chars: int = 60) -> str:
    """
    Extrae una keyword/título corto a partir del texto de la publicación.

    El texto de la publicación es lo que el autor escribió como caption del
    video, incluyendo hashtags. La estrategia es:
    - Priorizar los hashtags (#Starlink, #5G…) por ser etiquetas temáticas
      explícitas que el autor eligió para describir el contenido.
    - Si no hay hashtags útiles, tomar las primeras palabras significativas
      del caption (≥4 letras, no stopwords), conservando su orden de aparición.

    Args:
        descripcion: texto de la publicación (caption + hashtags).
        max_palabras: número máximo de palabras en la keyword resultante.
        max_chars: longitud máxima del título resultante.

    Returns:
        Keyword/título como string, o "" si no hay texto aprovechable.
    """
    if not descripcion:
        return ""

    import unicodedata as _ud

    # 1. Hashtags: el autor los usa como etiquetas temáticas explícitas.
    hashtags = re.findall(r'#(\w+)', descripcion)
    if hashtags:
        seleccion = []
        vistos = set()
        for tag in hashtags:
            low = tag.lower()
            if low in vistos or low in _STOPWORDS_TITULO:
                continue
            vistos.add(low)
            seleccion.append(tag.capitalize() if tag.islower() else tag)
            if len(seleccion) >= max_palabras:
                break
        if seleccion:
            return " ".join(seleccion)[:max_chars].strip()

    # 2. Sin hashtags útiles: primeras palabras significativas del caption.
    texto = re.sub(r'(https?://|www\.)\S+', '', descripcion)
    texto = re.sub(r'@\w+', '', texto)
    texto = "".join(
        c for c in texto
        if _ud.category(c) not in ("So", "Sm", "Sk", "Sc")
    )

    palabras = re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{4,}\b', texto)
    seleccion = []
    vistos = set()
    for w in palabras:
        low = w.lower()
        if low in vistos or low in _STOPWORDS_TITULO:
            continue
        vistos.add(low)
        seleccion.append(low.capitalize())
        if len(seleccion) >= max_palabras:
            break

    return " ".join(seleccion)[:max_chars].strip()


def batch_transcribe(videos: list[dict], config: dict, output_dir: Path) -> list[dict]:
    """
    Transcribe una lista de videos y guarda los resultados.

    Args:
        videos: lista de dicts de metadata de video.
        config: configuracion del proyecto.
        output_dir: directorio donde guardar las transcripciones.

    Returns:
        Lista de dicts con transcripciones.
    """
    results = []
    for i, video in enumerate(videos):
        logger.info("Transcribiendo video %d/%d", i + 1, len(videos))
        transcript = get_transcript(video, config, output_dir)
        results.append(transcript)

    return results
