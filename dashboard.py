"""
Comment Analytics Dashboard - Streamlit Application
Dos vistas: Individual (por operador + video) y Global (comparativa).
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.word_clouds.phrase_cloud import build_phrase_cloud_figure
from src.word_clouds.topic_extractor import (
    build_topic_cloud_figure,
    generate_topic_json_for_operator,
    get_topic_frequencies,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Comment Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_ROOT = Path(__file__).parent
OUTPUTS_DIR       = _ROOT / "outputs"
OUTPUTS_FRONT_DIR = _ROOT / "outputs_front"

OPERATORS = ["claro", "entel", "movistar", "bitel"]

LABEL_COLORS = {
    "positivo":    "#22C55E",
    "negativo":    "#DC2626",
    "informativo": "#7C3AED",
    "neutral":     "#6B7280",
    "irrelevante": "#9CA3AF",
}

LABEL_ES = {
    "positivo":    "Positivo",
    "negativo":    "Negativo",
    "informativo": "Informativo",
    "neutral":     "Neutral",
    "irrelevante": "Irrelevante",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8F9FA;
    color: #1F2937;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.section-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.video-item {
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.08);
    background: #FFFFFF;
    margin-bottom: 6px;
    cursor: pointer;
    font-size: 0.88rem;
    color: #374151;
}
.video-item:hover { border-color: #3B82F6; }
.video-item.active {
    border-color: #3B82F6;
    background: #EFF6FF;
    color: #1D4ED8;
    font-weight: 600;
}

.stat-bar-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #374151;
    text-transform: capitalize;
}
.stat-bar-value {
    font-size: 0.8rem;
    color: #6B7280;
}

.global-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.global-operator {
    font-size: 1rem;
    font-weight: 700;
    color: #1F2937;
    margin-bottom: 10px;
    text-transform: capitalize;
}
.comment-row {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.comment-text { color: #374151; font-size: 0.9rem; line-height: 1.5; }
.comment-meta { color: #9CA3AF; font-size: 0.76rem; margin-top: 4px; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}
.dashboard-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1F2937;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_classified_videos(operador: str) -> list[dict]:
    """Carga todos los clasificados (TikTok y Facebook) de un operador."""
    folder = OUTPUTS_DIR / operador / "clasificado"
    if not folder.exists():
        return []
    videos = []
    for pattern in ("VIDEO_CLASSIFIED_*.json", "facebook_classified_*.json"):
        for f in sorted(folder.glob(pattern)):
            try:
                videos.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue
    return videos


@st.cache_data
def load_phrase_data(operador: str, item_id: str, platform: str = "tiktok") -> dict:
    """Carga el JSON de frases de un video/post específico según plataforma."""
    prefix = "VIDEO_PHRASES_" if platform == "tiktok" else "FB_PHRASES_"
    path = OUTPUTS_FRONT_DIR / operador / f"{prefix}{item_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data
def load_general_phrase_data(operador: str) -> dict:
    """Carga el JSON de frases generales del mes (TikTok + Facebook agregados)."""
    path = OUTPUTS_FRONT_DIR / operador / "GENERAL_PHRASES.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_item_id(video: dict) -> str:
    """Devuelve el ID del item independientemente de la plataforma."""
    return video.get("video_id") or video.get("post_id", "")


def _get_item_fecha(video: dict) -> str:
    """Devuelve la fecha del item independientemente de la plataforma."""
    return video.get("fecha_video") or video.get("fecha_post", "")


def _get_item_platform(video: dict) -> str:
    """Devuelve la plataforma del item: 'tiktok' o 'facebook'."""
    return video.get("plataforma", "tiktok")


def _get_video_label(video: dict, index: int) -> str:
    """Genera etiqueta corta para el item en la lista, con badge de plataforma."""
    fecha = _get_item_fecha(video)
    item_id = _get_item_id(video)
    platform = _get_item_platform(video)
    badge = "TT" if platform == "tiktok" else "FB"
    return f"[{badge}] {index + 1}  ·  {fecha}  ·  …{item_id[-6:]}"


# ---------------------------------------------------------------------------
# Componentes
# ---------------------------------------------------------------------------
def render_stat_bars(counts: dict[str, int], total: int):
    """Barras de progreso por label con color."""
    labels_order = ["negativo", "positivo", "neutral", "informativo", "irrelevante"]
    rows = ""
    for lbl in labels_order:
        count = counts.get(lbl, 0)
        if count == 0:
            continue
        pct = count / total * 100 if total else 0
        color = LABEL_COLORS.get(lbl, "#6B7280")
        label_es = LABEL_ES.get(lbl, lbl.capitalize())
        rows += f"""
<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span class="stat-bar-label">{label_es}</span>
    <span class="stat-bar-value">{count} &nbsp;<b style="color:{color};">{pct:.1f}%</b></span>
  </div>
  <div style="background:rgba(0,0,0,0.06);border-radius:6px;height:8px;overflow:hidden;">
    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:6px;"></div>
  </div>
</div>"""
    st.markdown(rows, unsafe_allow_html=True)


def render_phrase_cloud_section(operador: str, item_id: str, comments: list[dict], platform: str = "tiktok"):
    """Tabs positivo/negativo con nube de frases para un video/post específico."""
    phrase_data = load_phrase_data(operador, item_id, platform=platform)

    tab_pos, tab_neg = st.tabs(["☀ Positivo", "⚡ Negativo"])

    with tab_pos:
        freqs_pos = phrase_data.get("positivo", {})
        if freqs_pos:
            fig = build_phrase_cloud_figure(comments, "positivo")
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Sin frases positivas.")

    with tab_neg:
        freqs_neg = phrase_data.get("negativo", {})
        if freqs_neg:
            fig = build_phrase_cloud_figure(comments, "negativo")
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Sin frases negativas.")


def _render_wordcloud_from_freqs(freqs: dict, sentiment: str):
    """Pinta una wordcloud dado un dict {texto: frecuencia}."""
    from wordcloud import WordCloud as WC
    if not freqs:
        return False
    wc = WC(
        width=1000, height=500, background_color="white",
        color_func=lambda *_, **__: LABEL_COLORS[sentiment],
        stopwords=set(), max_words=50, collocations=False,
        prefer_horizontal=0.85, min_word_length=2,
    ).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return True


def render_general_phrase_cloud(operador: str):
    """Nube de frases/topics con toggle. Topics requieren GENERAL_TOPICS.json generado."""
    topics_path = OUTPUTS_FRONT_DIR / operador / "GENERAL_TOPICS.json"
    has_topics = topics_path.exists()

    # toggle solo visible si hay JSON de topics
    if has_topics:
        modo = st.radio(
            "Modo nube",
            options=["Topics (IA)", "Frases"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"nube_modo_{operador}",
        )
    else:
        modo = "Frases"
        st.caption("💡 Genera topics con IA ejecutando el pipeline de topics.")

    tab_pos, tab_neg = st.tabs(["☀ Positivo", "⚡ Negativo"])

    if modo == "Topics (IA)":
        with tab_pos:
            freqs = get_topic_frequencies(operador, "positivo")
            if not freqs:
                st.info("Sin topics positivos generados.")
            else:
                if not _render_wordcloud_from_freqs(freqs, "positivo"):
                    st.info("Sin topics positivos.")
                # lista de topics con frecuencia
                try:
                    data = json.loads(topics_path.read_text(encoding="utf-8"))
                    topics = data.get("positivo", {})
                    for name, info in sorted(topics.items(), key=lambda x: -x[1]["frecuencia"]):
                        st.markdown(
                            f'<span style="font-size:0.8rem;color:#6B7280;">'
                            f'<b style="color:#22C55E;">{info["frecuencia"]}</b>&nbsp; {name}</span>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass

        with tab_neg:
            freqs = get_topic_frequencies(operador, "negativo")
            if not freqs:
                st.info("Sin topics negativos generados.")
            else:
                if not _render_wordcloud_from_freqs(freqs, "negativo"):
                    st.info("Sin topics negativos.")
                try:
                    data = json.loads(topics_path.read_text(encoding="utf-8"))
                    topics = data.get("negativo", {})
                    for name, info in sorted(topics.items(), key=lambda x: -x[1]["frecuencia"]):
                        st.markdown(
                            f'<span style="font-size:0.8rem;color:#6B7280;">'
                            f'<b style="color:#DC2626;">{info["frecuencia"]}</b>&nbsp; {name}</span>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass

    else:  # modo Frases
        phrase_data = load_general_phrase_data(operador)
        if not phrase_data:
            st.info("Sin datos de frases. Ejecuta el pipeline primero.")
            return

        with tab_pos:
            freqs = phrase_data.get("positivo", {})
            if not _render_wordcloud_from_freqs(freqs, "positivo"):
                st.info("Sin frases positivas en el mes.")

        with tab_neg:
            freqs = phrase_data.get("negativo", {})
            if not _render_wordcloud_from_freqs(freqs, "negativo"):
                st.info("Sin frases negativas en el mes.")


def render_region_chart(comments: list[dict]):
    """Barras horizontales de distribución por departamento (solo comentarios con región identificada)."""
    from collections import Counter
    deptos = [
        c.get("departamento", "No especificado")
        for c in comments
        if c.get("departamento", "No especificado") != "No especificado"
    ]

    if not deptos:
        st.caption("Sin comentarios con región identificada.")
        return

    counts = Counter(deptos)
    total = sum(counts.values())
    palette = [
        "#3B82F6", "#22C55E", "#DC2626", "#7C3AED",
        "#F59E0B", "#06B6D4", "#EC4899", "#84CC16",
        "#F97316", "#14B8A6", "#A855F7", "#6B7280",
    ]

    rows = ""
    for i, (depto, count) in enumerate(counts.most_common()):
        pct = count / total * 100
        color = palette[i % len(palette)]
        rows += f"""
<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
    <span style="font-size:0.82rem;font-weight:600;color:#374151;">{depto}</span>
    <span style="font-size:0.8rem;color:#6B7280;">{count} &nbsp;<b style="color:{color};">{pct:.1f}%</b></span>
  </div>
  <div style="background:rgba(0,0,0,0.06);border-radius:6px;height:7px;overflow:hidden;">
    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:6px;"></div>
  </div>
</div>"""

    st.markdown(rows, unsafe_allow_html=True)
    st.caption(f"{len(deptos)} de {len(comments)} comentarios con región identificada")


COMMENTS_PER_PAGE = 8
_LABELS_ORDER = ["negativo", "positivo", "informativo", "neutral", "irrelevante"]


def render_comments_by_label(comments: list[dict], video_id: str, operador: str):
    """Expanders por clasificación, con paginación interna en cada uno."""
    from collections import Counter
    import html as html_lib

    by_label: dict[str, list[dict]] = {lbl: [] for lbl in _LABELS_ORDER}
    for c in comments:
        lbl = c.get("clasificacion", "neutral")
        if lbl in by_label:
            by_label[lbl].append(c)

    for lbl in _LABELS_ORDER:
        items = by_label[lbl]
        if not items:
            continue

        color  = LABEL_COLORS.get(lbl, "#6B7280")
        label_es = LABEL_ES.get(lbl, lbl.capitalize())
        header = f'<span style="color:{color};font-weight:700;">{label_es}</span> — {len(items)} comentarios'

        with st.expander(f"{label_es} ({len(items)})", expanded=(lbl == "negativo")):
            # clave de página única por operador + video + label
            page_key = f"page_{operador}_{video_id}_{lbl}"
            if page_key not in st.session_state:
                st.session_state[page_key] = 0

            total_pages = max(1, (len(items) + COMMENTS_PER_PAGE - 1) // COMMENTS_PER_PAGE)
            current_page = min(st.session_state[page_key], total_pages - 1)
            start = current_page * COMMENTS_PER_PAGE
            page_items = items[start: start + COMMENTS_PER_PAGE]

            for c in page_items:
                text  = html_lib.escape(str(c.get("comentario", "")))
                fecha = str(c.get("fecha", ""))[:16]
                st.markdown(
                    f"""<div class="comment-row">
  <div class="comment-text">{text}</div>
  <div class="comment-meta">{fecha}</div>
</div>""",
                    unsafe_allow_html=True,
                )

            # paginación
            if total_pages > 1:
                p_prev, p_info, p_next = st.columns([1, 2, 1])
                with p_prev:
                    if st.button("← Ant", key=f"prev_{page_key}", disabled=(current_page == 0)):
                        st.session_state[page_key] = current_page - 1
                        st.rerun()
                with p_info:
                    st.markdown(
                        f'<div style="text-align:center;color:#6B7280;font-size:0.8rem;padding-top:6px;">'
                        f'Pág. {current_page + 1} / {total_pages}</div>',
                        unsafe_allow_html=True,
                    )
                with p_next:
                    if st.button("Sig →", key=f"next_{page_key}", disabled=(current_page >= total_pages - 1)):
                        st.session_state[page_key] = current_page + 1
                        st.rerun()


def _periodo_from_fecha(fecha: str) -> str:
    """Extrae YYYYMM de una fecha 'YYYY-MM-DD' o 'YYYY-MM-...'."""
    if not fecha or len(fecha) < 7:
        return "000000"
    return fecha[:4] + fecha[5:7]


@st.cache_data
def load_snps_by_periodo() -> dict[str, dict[str, dict]]:
    """
    Devuelve { periodo: { operador: {total, positivo, negativo, neutral,
                                     informativo, irrelevante, snps} } }
    El periodo se deriva de fecha_video / fecha_post (YYYYMM).
    """
    from collections import Counter, defaultdict

    # { operador: { periodo: Counter } }
    data: dict[str, dict[str, Counter]] = {op: defaultdict(Counter) for op in OPERATORS}

    for op in OPERATORS:
        for video in load_classified_videos(op):
            fecha = _get_item_fecha(video)
            periodo = _periodo_from_fecha(fecha)
            if periodo == "000000":
                continue
            for c in video.get("comentarios", []):
                lbl = c.get("clasificacion", "neutral")
                data[op][periodo][lbl] += 1

    # Construir resultado final
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for op in OPERATORS:
        for periodo, counts in data[op].items():
            total = sum(counts.values())
            pos   = counts.get("positivo", 0)
            neg   = counts.get("negativo", 0)
            snps  = ((pos - neg) / total * 100) if total else 0.0
            result[periodo][op] = {
                "total":        total,
                "positivo":     pos,
                "negativo":     neg,
                "neutral":      counts.get("neutral", 0),
                "informativo":  counts.get("informativo", 0),
                "irrelevante":  counts.get("irrelevante", 0),
                "snps":         snps,
            }

    return dict(result)


OP_COLORS = {
    "claro":    "#DC2626",
    "entel":    "#2563EB",
    "movistar": "#059669",
    "bitel":    "#7C3AED",
}


def render_global_mockup():
    """Vista global — sNPS real por operador + gráfico de tendencia por periodo."""
    st.markdown("### Comparativa de Operadores — sNPS")

    snps_data = load_snps_by_periodo()

    if not snps_data:
        st.info("Sin datos clasificados. Ejecuta el pipeline primero.")
        return

    periodos = sorted(snps_data.keys())
    ultimo   = periodos[-1]

    # Formato legible: 202605 → "May 2026"
    import calendar
    def fmt_periodo(p: str) -> str:
        try:
            y, m = int(p[:4]), int(p[4:])
            return f"{calendar.month_abbr[m]} {y}"
        except Exception:
            return p

    # ── Métricas del último periodo ──────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:0.85rem;color:#6B7280;margin-bottom:8px;">'
        f'Periodo actual: <b>{fmt_periodo(ultimo)}</b></div>',
        unsafe_allow_html=True,
    )

    ultimo_data = snps_data[ultimo]
    sorted_ops  = sorted(
        [(op, ultimo_data.get(op, {})) for op in OPERATORS],
        key=lambda x: x[1].get("snps", -999),
        reverse=True,
    )

    cols = st.columns(len(OPERATORS))
    for col, (op, d) in zip(cols, sorted_ops):
        snps_val = d.get("snps", 0.0)
        total    = d.get("total", 0)
        color    = OP_COLORS[op]
        with col:
            st.markdown(
                f"""<div style="background:#fff;border:1px solid rgba(0,0,0,0.08);
                border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);
                text-align:center;">
                  <div style="font-size:0.9rem;font-weight:700;color:{color};
                       text-transform:capitalize;margin-bottom:6px;">{op.capitalize()}</div>
                  <div style="font-size:2rem;font-weight:800;
                       color:{'#22C55E' if snps_val >= 0 else '#DC2626'};">
                    {snps_val:+.1f}
                  </div>
                  <div style="font-size:0.7rem;color:#6B7280;margin-top:2px;">sNPS</div>
                  <div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">
                    {total:,} comentarios
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tarjetas detalle del último periodo ──────────────────────────────────
    for op, d in sorted_ops:
        total = d.get("total", 0)
        if total == 0:
            continue
        pos_pct  = d["positivo"]    / total * 100
        neg_pct  = d["negativo"]    / total * 100
        neu_pct  = d["neutral"]     / total * 100
        info_pct = d["informativo"] / total * 100
        irr_pct  = d["irrelevante"] / total * 100
        snps_val  = d["snps"]
        snps_color = "#22C55E" if snps_val >= 0 else "#DC2626"
        op_color   = OP_COLORS[op]

        bar_seg = ""
        for lbl, pct in [("positivo", pos_pct), ("negativo", neg_pct),
                         ("neutral", neu_pct), ("informativo", info_pct),
                         ("irrelevante", irr_pct)]:
            if pct > 0:
                bar_seg += (
                    f'<div style="width:{pct:.2f}%;height:100%;'
                    f'background:{LABEL_COLORS[lbl]};"></div>'
                )

        st.markdown(f"""
<div class="global-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div class="global-operator" style="color:{op_color};">{op.capitalize()}</div>
    <div style="font-size:1.35rem;font-weight:800;color:{snps_color};">
      sNPS&nbsp;{snps_val:+.1f}
    </div>
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
    <span class="badge" style="background:{LABEL_COLORS['positivo']};">{d['positivo']} pos ({pos_pct:.1f}%)</span>
    <span class="badge" style="background:{LABEL_COLORS['negativo']};">{d['negativo']} neg ({neg_pct:.1f}%)</span>
    <span class="badge" style="background:{LABEL_COLORS['neutral']};">{d['neutral']} neu ({neu_pct:.1f}%)</span>
    <span class="badge" style="background:{LABEL_COLORS['informativo']};">{d['informativo']} inf ({info_pct:.1f}%)</span>
    <span class="badge" style="background:{LABEL_COLORS['irrelevante']};">{d['irrelevante']} irr ({irr_pct:.1f}%)</span>
  </div>
  <div style="display:flex;height:12px;border-radius:8px;overflow:hidden;gap:1px;">{bar_seg}</div>
  <div style="font-size:0.75rem;color:#9CA3AF;margin-top:6px;">
    Total: {total:,} comentarios (TikTok + Facebook)
  </div>
</div>""", unsafe_allow_html=True)

    # ── Gráfico de líneas (solo si hay más de un periodo) ────────────────────
    if len(periodos) > 1:
        st.divider()
        st.markdown("### Evolución del sNPS por periodo")

        labels_x = [fmt_periodo(p) for p in periodos]

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#F8F9FA")
        ax.set_facecolor("#F8F9FA")

        for op in OPERATORS:
            snps_series = [
                snps_data[p].get(op, {}).get("snps", None)
                for p in periodos
            ]
            # Trazar solo los puntos que tienen datos
            x_vals, y_vals = [], []
            for i, v in enumerate(snps_series):
                if v is not None:
                    x_vals.append(i)
                    y_vals.append(v)
            if not x_vals:
                continue
            ax.plot(
                x_vals, y_vals,
                marker="o", linewidth=2.5, markersize=7,
                color=OP_COLORS[op], label=op.capitalize(),
            )
            # Etiqueta en el último punto
            ax.annotate(
                f"{y_vals[-1]:+.1f}",
                (x_vals[-1], y_vals[-1]),
                textcoords="offset points", xytext=(6, 0),
                fontsize=8, color=OP_COLORS[op], fontweight="bold",
            )

        ax.axhline(0, color="#D1D5DB", linewidth=1, linestyle="--")
        ax.set_xticks(range(len(periodos)))
        ax.set_xticklabels(labels_x, fontsize=9)
        ax.set_ylabel("sNPS", fontsize=9, color="#6B7280")
        ax.tick_params(colors="#6B7280", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#E5E7EB")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(pad=1.5)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Vista individual
# ---------------------------------------------------------------------------
def render_individual():
    # Selector de operador
    op_tabs = st.tabs([op.capitalize() for op in OPERATORS])

    for i, operador in enumerate(OPERATORS):
        with op_tabs[i]:
            videos = load_classified_videos(operador)

            if not videos:
                st.info(f"Sin datos clasificados para {operador.capitalize()}. "
                        f"Ejecuta `run_classify_from_files('{operador}')` primero.")
                continue

            # Filtro de plataforma
            plat_key = f"platform_filter_{operador}"
            has_tt = any(_get_item_platform(v) == "tiktok"   for v in videos)
            has_fb = any(_get_item_platform(v) == "facebook" for v in videos)
            plat_options = []
            if has_tt: plat_options.append("TikTok")
            if has_fb: plat_options.append("Facebook")

            if len(plat_options) > 1:
                plat_sel = st.radio(
                    "Plataforma",
                    options=plat_options,
                    horizontal=True,
                    key=plat_key,
                    label_visibility="collapsed",
                )
            else:
                plat_sel = plat_options[0] if plat_options else "TikTok"
                st.caption(f"Plataforma: {plat_sel}")

            plat_filter = "tiktok" if plat_sel == "TikTok" else "facebook"
            filtered = [v for v in videos if _get_item_platform(v) == plat_filter]

            # Estado de ítem seleccionado — se resetea al cambiar de plataforma
            state_key  = f"selected_video_{operador}"
            plat_prev_key = f"platform_prev_{operador}"
            if st.session_state.get(plat_prev_key) != plat_filter:
                st.session_state[state_key] = 0
                st.session_state[plat_prev_key] = plat_filter
            if state_key not in st.session_state:
                st.session_state[state_key] = 0

            # Layout: lista lateral | stats | nube
            list_col, stats_col, cloud_col = st.columns([1, 1, 2], gap="large")

            with list_col:
                st.markdown(f"**{'Videos' if plat_filter == 'tiktok' else 'Posts'} ({len(filtered)})**")
                for idx, video in enumerate(filtered):
                    is_active = st.session_state[state_key] == idx
                    label = _get_video_label(video, idx)
                    if st.button(
                        label,
                        key=f"vid_{operador}_{plat_filter}_{idx}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state[state_key] = idx
                        st.rerun()

            selected_idx = min(st.session_state[state_key], len(filtered) - 1)
            video = filtered[selected_idx]
            comments = video.get("comentarios", [])
            total = video.get("total_comentarios", len(comments))
            item_id = _get_item_id(video)
            item_fecha = _get_item_fecha(video)
            platform = _get_item_platform(video)
            platform_label = "TikTok" if platform == "tiktok" else "Facebook"

            from collections import Counter
            counts = dict(Counter(c.get("clasificacion", "neutral") for c in comments))

            with stats_col:
                st.markdown(f"**Estadísticas · {platform_label}**")
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#6B7280;margin-bottom:12px;">'
                    f'{item_fecha} &nbsp;·&nbsp; {total} comentarios</div>',
                    unsafe_allow_html=True,
                )
                render_stat_bars(counts, total)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Distribución por región**")
                render_region_chart(comments)

            with cloud_col:
                st.markdown("**Nube de frases**")
                render_phrase_cloud_section(operador, item_id, comments, platform=platform)

            st.divider()
            st.markdown("**Comentarios**")
            render_comments_by_label(comments, item_id, operador)

            st.divider()
            st.markdown("**Nube de frases general del mes (TikTok + Facebook)**")
            render_general_phrase_cloud(operador)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_css()

    st.markdown('<div class="dashboard-title">Comment Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    vista = st.radio(
        "Vista",
        options=["Individual", "Global"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if vista == "Individual":
        render_individual()
    else:
        render_global_mockup()


if __name__ == "__main__":
    main()
