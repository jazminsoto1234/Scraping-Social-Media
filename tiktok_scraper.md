<style>
  body, .vscode-body {
    background: #ffffff !important;
    color: #000000 !important;
  }
  .vscode-body a {
    color: #0645ad !important;
  }
  .vscode-body code {
    background: #f5f5f5 !important;
  }
</style>

# Scraper de comentarios TikTok

Este script descarga comentarios y replies de un video de TikTok usando Playwright y una sesion ya iniciada en Chrome. La entrada principal es la URL del video definida en `VIDEO_URL` y el resultado es un JSON simplificado con comentarios + replies.

Archivo analizado: [src/scrapers/tiktok/scraper.py](src/scrapers/tiktok/scraper.py#L1-L256)

## Flujo general

1. Conecta a un Chrome real via CDP en `http://localhost:9222`.
2. Busca una pestaña que ya tenga el video abierto; si no existe, pide al usuario abrir la URL.
3. Intercepta respuestas de red para capturar `api/comment/list` y `api/comment/list/reply`.
4. Hace click en el tab de comentarios si es necesario y detecta el panel scrolleable.
5. Scrollea en bloques con delays aleatorios para cargar comentarios padre.
6. Reintenta y recalibra el panel si no hay cambios.
7. Vuelve al tope y expande replies clickeando botones visibles.
8. Guarda incrementalmente el JSON durante todo el proceso y al final.

## Entradas y dependencias

- `VIDEO_URL`: URL del video a procesar. (Cambiar aqui para otro video.)
- Chrome debe estar abierto con `--remote-debugging-port=9222` y una sesion valida.
- Usa `playwright.sync_api`.

## Salidas

- Archivo JSON: `video_<VIDEO_ID>_comments.json` en el directorio de trabajo.
- Log de debug: `cuarto_video_debug.log`.

## Estructura del JSON de salida (simplificado)

```json
{
  "video": { "id": "...", "autor": "...", "url": "..." },
  "total_comentarios": 123,
  "total_respuestas": 456,
  "total_items": 579,
  "comentarios": [
    {
      "id": "...",
      "usuario_id": "...",
      "nickname": "...",
      "comentario": "...",
      "likes": 0,
      "fecha": "YYYY-MM-DD HH:MM:SS",
      "num_respuestas": 0,
      "respuestas": [
        {
          "id": "...",
          "usuario_id": "...",
          "nickname": "...",
          "comentario": "...",
          "likes": 0,
          "fecha": "YYYY-MM-DD HH:MM:SS"
        }
      ]
    }
  ]
}
```

## Notas relevantes

- El docstring dice que guarda en `test/`, pero el codigo actual escribe en el directorio de ejecucion.
- La variable `STORAGE_FILE` esta definida pero no se usa.
- El script hace scroll "humano" con pausas aleatorias para reducir bloqueos.

## Ejecucion rapida

Ejecuta el script con el entorno configurado y Chrome abierto con remote debugging:

```bash
python3 src/scrapers/tiktok/scraper.py
```
