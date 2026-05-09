import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cluster_commands import build_dbscan_command, build_hdbscan_command, build_kmeans_command
from kmeans_text_cluster import clean_text, clean_text_for_embedding, embedding_preprocessing_steps
from project_paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR, RESOURCES_DIR, ensure_parent_dir


st.set_page_config(page_title="Text Clustering Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #dbe4f0;
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 132px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .metric-label {
        color: #475569;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.15;
        word-break: break-word;
    }
    .metric-hint {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


HISTORY_PATH = REPORTS_DIR / "run_history.json"
GENERAL_HISTORY_FIELDS = [
    "input",
    "text_col",
    "min_len",
    "min_words",
    "limit",
    "limit_random",
    "drop_duplicate_texts",
    "batch_size",
    "sample_size",
    "embedding_model",
    "device",
    "keep_hashtags",
    "remove_emojis",
    "stopwords",
    "no_default_stopwords",
]
MODEL_HISTORY_FIELDS = {
    "KMeans": ["k", "kmeans_output", "kmeans_report", "kmeans_extra"],
    "DBSCAN": [
        "eps",
        "min_samples",
        "dbscan_use_umap",
        "dbscan_umap_components",
        "dbscan_umap_neighbors",
        "dbscan_umap_min_dist",
        "dbscan_output",
        "dbscan_report",
        "dbscan_extra",
    ],
    "HDBSCAN": [
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_epsilon",
        "hdbscan_use_umap",
        "hdbscan_umap_components",
        "hdbscan_umap_neighbors",
        "hdbscan_umap_min_dist",
        "hdbscan_output",
        "hdbscan_report",
        "hdbscan_extra",
    ],
}
MODEL_PATH_FIELDS = {
    "KMeans": ("kmeans_output", "kmeans_report"),
    "DBSCAN": ("dbscan_output", "dbscan_report"),
    "HDBSCAN": ("hdbscan_output", "hdbscan_report"),
}


def csv_candidates() -> list[str]:
    patterns = [DATA_RAW_DIR.glob("*.csv"), (PROJECT_ROOT / "data" / "sample").glob("*.csv")]
    files: list[str] = []
    for pattern in patterns:
        files.extend(str(path) for path in sorted(pattern))
    return files


def load_history() -> list[dict[str, object]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(history, list):
        return []
    return [item for item in history if isinstance(item, dict)]


def save_history(history: list[dict[str, object]]) -> None:
    ensure_parent_dir(HISTORY_PATH)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=True), encoding="utf-8")


def pick_config_fields(config: dict[str, object], keys: list[str]) -> dict[str, object]:
    return {key: config.get(key) for key in keys}


def build_history_record(label: str, command: list[str], config: dict[str, object]) -> dict[str, object]:
    output_key, report_key = MODEL_PATH_FIELDS[label]
    output_path = str(config[output_key])
    report_path = str(config[report_key])
    report_text = load_report_text(report_path)
    report_metrics = parse_report_metrics(report_text)
    result_df, summary = summarize_result(output_path)
    timestamp = datetime.now().isoformat(timespec="seconds")
    return {
        "id": f"{timestamp}_{label.lower()}",
        "timestamp": timestamp,
        "model": label,
        "input": str(config["input"]),
        "text_col": str(config["text_col"]),
        "command": command,
        "output_csv": output_path,
        "report_path": report_path,
        "general_settings": pick_config_fields(config, GENERAL_HISTORY_FIELDS),
        "model_settings": pick_config_fields(config, MODEL_HISTORY_FIELDS[label]),
        "metrics": report_metrics,
        "summary": summary,
        "report_text": report_text,
        "result_preview": result_preview_rows(result_df),
    }


def append_history_record(record: dict[str, object]) -> None:
    history = load_history()
    history.insert(0, record)
    save_history(history)


def history_table(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        metrics = record.get("metrics", {})
        summary = record.get("summary", {})
        rows.append(
            {
                "No": idx,
                "Waktu": record.get("timestamp", "-"),
                "Model": record.get("model", "-"),
                "Input": record.get("input", "-"),
                "Silhouette": metrics.get("Silhouette", "-"),
                "Davies-Bouldin": metrics.get("Davies-Bouldin", "-"),
                "Clusters": metrics.get("Clusters", summary.get("clusters", "-")),
                "Noise Rate": metrics.get("Noise Rate", "-"),
            }
        )
    return pd.DataFrame(rows)


def render_history_detail(record: dict[str, object]) -> None:
    st.markdown("**Detail Run**")
    download_name = (
        f"history_{record.get('model', 'model').lower()}_"
        f"{str(record.get('timestamp', 'run')).replace(':', '-')}"
        ".json"
    )
    st.download_button(
        "Download Detail",
        data=json.dumps(record, indent=2, ensure_ascii=True),
        file_name=download_name,
        mime="application/json",
        use_container_width=True,
    )

    meta_df = pd.DataFrame(
        [
            {"Field": "Waktu", "Value": record.get("timestamp", "-")},
            {"Field": "Model", "Value": record.get("model", "-")},
            {"Field": "Input", "Value": record.get("input", "-")},
            {"Field": "Kolom Teks", "Value": record.get("text_col", "-")},
            {"Field": "Output CSV", "Value": record.get("output_csv", "-")},
            {"Field": "Report Path", "Value": record.get("report_path", "-")},
        ]
    )
    st.dataframe(meta_df, hide_index=True, use_container_width=True)

    metrics = record.get("metrics", {})
    summary = record.get("summary", {})
    metric_rows = [{"Metric": key, "Value": value} for key, value in metrics.items()]
    metric_rows.extend(
        [
            {"Metric": "rows", "Value": summary.get("rows", "-")},
            {"Metric": "clusters", "Value": summary.get("clusters", "-")},
            {"Metric": "noise", "Value": summary.get("noise", "-")},
            {"Metric": "filtered_out", "Value": summary.get("filtered_out", "-")},
        ]
    )
    st.markdown("**Metrics**")
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)

    st.markdown("**General Settings**")
    st.dataframe(
        pd.DataFrame(
            [{"Setting": key, "Value": value} for key, value in record.get("general_settings", {}).items()]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Model Settings**")
    st.dataframe(
        pd.DataFrame(
            [{"Setting": key, "Value": value} for key, value in record.get("model_settings", {}).items()]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Command**")
    st.code(" ".join(record.get("command", [])))

    st.markdown("**Result Preview**")
    result_preview = record.get("result_preview", [])
    if result_preview:
        st.dataframe(pd.DataFrame(result_preview), use_container_width=True)
    else:
        st.write("Result preview tidak tersedia.")

    st.markdown("**Detail Report**")
    report_text = str(record.get("report_text", "")).strip()
    if report_text:
        st.code(report_text)
    else:
        st.write("Detail report tidak tersedia.")


def render_history_page() -> None:
    st.title("History Run")
    st.caption("Lihat riwayat skor dan setting dari setiap run model.")

    records = load_history()
    if not records:
        st.info("Belum ada history run.")
        return

    model_options = ["Semua", "KMeans", "DBSCAN", "HDBSCAN"]
    selected_model = st.selectbox("Filter model", options=model_options)
    filtered_records = records if selected_model == "Semua" else [record for record in records if record.get("model") == selected_model]

    if not filtered_records:
        st.info("Belum ada history untuk filter yang dipilih.")
        return

    st.dataframe(history_table(filtered_records), hide_index=True, use_container_width=True)

    record_map = {str(record.get("id", "")): record for record in filtered_records}
    selected_record_id = st.selectbox(
        "Pilih history",
        options=list(record_map),
        format_func=lambda record_id: (
            f"{record_map[record_id].get('timestamp', '-')} | "
            f"{record_map[record_id].get('model', '-')} | "
            f"{Path(str(record_map[record_id].get('input', '-'))).name}"
        ),
    )

    if st.button("Lihat Detail", use_container_width=True):
        st.session_state["selected_history_id"] = selected_record_id

    selected_id = st.session_state.get("selected_history_id")
    if selected_id:
        detail_record = next((record for record in records if record.get("id") == selected_id), None)
        if detail_record is not None:
            render_history_detail(detail_record)


def detect_columns(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception:
        return []
    return list(df.columns)


def build_preprocessing_preview(
    path: str,
    text_col: str,
    *,
    rows: int,
    remove_hashtags: bool,
    remove_emojis: bool,
) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=[text_col], nrows=rows)
    except Exception:
        return None

    if text_col not in df.columns:
        return None

    preview = df.copy()
    preview[text_col] = preview[text_col].fillna("").astype(str)
    preview = preview.rename(columns={text_col: "raw_text"})
    preview["clean_text"] = preview["raw_text"].map(clean_text)
    step_rows = preview["raw_text"].map(
        lambda text: embedding_preprocessing_steps(
            text,
            remove_hashtags=remove_hashtags,
            remove_emojis=remove_emojis,
        )
    ).tolist()
    step_df = pd.DataFrame(step_rows)
    step_df = step_df.rename(
        columns={
            "raw_text": "embedding_raw_text",
            "unicode_normalized": "step_1_unicode_normalized",
            "whitespace_normalized": "step_2_whitespace_normalized",
            "url_normalized": "step_3_url_normalized",
            "mention_normalized": "step_4_mention_normalized",
            "hashtag_normalized": "step_5_hashtag_normalized",
            "dash_separator_normalized": "step_6_dash_separator_normalized",
            "emoji_normalized": "step_7_emoji_normalized",
            "final_text": "step_8_final_embedding_text",
        }
    )
    return pd.concat([preview, step_df.drop(columns=["embedding_raw_text"])], axis=1)


def run_command(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout


def load_report_text(path: str) -> str:
    report_path = Path(path)
    if not report_path.exists():
        return ""
    return report_path.read_text(encoding="utf-8")


def parse_report_metrics(report_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not report_text:
        return metrics

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("==="):
            continue

        if line.startswith("Rows used:"):
            metrics["Rows Used"] = line.removeprefix("Rows used:").strip()
        elif line.startswith("embedding_model="):
            metrics["Embedding Model"] = line.split("=", 1)[1].strip()
        elif line.startswith("k="):
            metrics["KMeans Config"] = line
        elif line.startswith("eps="):
            metrics["DBSCAN Config"] = line
        elif line.startswith("min_cluster_size="):
            metrics["HDBSCAN Config"] = line
        elif line.startswith("clusters="):
            match = re.search(r"clusters=(\d+)\s+noise_points=(\d+)\s+noise_rate=([0-9.]+)", line)
            if match:
                metrics["Clusters"] = match.group(1)
                metrics["Noise Points"] = match.group(2)
                metrics["Noise Rate"] = f"{float(match.group(3)):.2%}"
        elif line.startswith("filter_passes="):
            metrics["Filter Passes"] = line
        elif line.startswith("inertia="):
            metrics["Inertia"] = line.split("=", 1)[1].strip()
        elif line.startswith("silhouette_cosine"):
            metrics["Silhouette"] = line.rsplit("=", 1)[1].strip()
        elif line.startswith("davies_bouldin"):
            metrics["Davies-Bouldin"] = line.split("=", 1)[1].split()[0].strip()

    return metrics


def summarize_result(path: str) -> tuple[pd.DataFrame | None, dict[str, int | float]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None, {}

    df = pd.read_csv(csv_path)
    if "cluster" not in df.columns:
        return df, {"rows": len(df)}

    clusters = pd.to_numeric(df["cluster"], errors="coerce")
    valid_clusters = clusters.dropna()
    non_noise = valid_clusters[valid_clusters != -1]
    noise_count = int((valid_clusters == -1).sum())
    filtered_out = 0
    if "filtered_out" in df.columns:
        filtered_out = int(df["filtered_out"].fillna(False).astype(bool).sum())

    summary = {
        "rows": int(len(df)),
        "clusters": int(non_noise.nunique()),
        "noise": noise_count,
        "filtered_out": filtered_out,
    }
    if len(df):
        summary["noise_rate"] = float(noise_count / len(df))
    return df, summary


def result_preview_rows(df: pd.DataFrame | None, *, limit: int = 100) -> list[dict[str, object]]:
    if df is None or df.empty:
        return []
    preview_df = df.head(limit).copy()
    preview_df = preview_df.where(pd.notna(preview_df), None)
    return preview_df.to_dict(orient="records")


def cluster_chart_data(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or "cluster" not in df.columns:
        return pd.DataFrame(columns=["cluster", "count"])
    counts = df["cluster"].value_counts(dropna=False).sort_index()
    chart_df = counts.rename_axis("cluster").reset_index(name="count")
    chart_df["cluster"] = chart_df["cluster"].astype(str)
    return chart_df


def render_metric_cards(report_metrics: dict[str, str]) -> None:
    cards = [
        ("Silhouette", report_metrics.get("Silhouette", "-"), "Range -1 to 1, closer to 1 better"),
        ("Davies-Bouldin", report_metrics.get("Davies-Bouldin", "-"), "Range 0+, lower better"),
    ]

    cols = st.columns(2)
    for col, (label, value, hint) in zip(cols, cards, strict=False):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-hint">{hint}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(title: str, output_path: str, report_path: str) -> None:
    st.subheader(title)
    df, summary = summarize_result(output_path)
    report_text = load_report_text(report_path)
    report_metrics = parse_report_metrics(report_text)
    if df is None:
        st.info(f"Belum ada output di `{output_path}`")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", summary.get("rows", 0))
    col2.metric("Clusters", summary.get("clusters", 0))
    col3.metric("Noise", summary.get("noise", 0))
    col4.metric("Filtered Out", summary.get("filtered_out", 0))
    if "noise_rate" in summary:
        st.caption(f"Noise rate: {summary['noise_rate']:.2%}")

    if report_metrics:
        st.markdown("**Metrics**")
        render_metric_cards(report_metrics)

        info_items = []
        for key in [
            "Rows Used",
            "Embedding Model",
            "KMeans Config",
            "DBSCAN Config",
            "HDBSCAN Config",
            "Clusters",
            "Noise Points",
            "Noise Rate",
            "Filter Passes",
            "Inertia",
        ]:
            if key in report_metrics:
                info_items.append({"Metric": key, "Value": report_metrics[key]})
        if info_items:
            st.dataframe(pd.DataFrame(info_items), hide_index=True, use_container_width=True)

    chart_df = cluster_chart_data(df)
    if not chart_df.empty:
        st.bar_chart(chart_df.set_index("cluster"))

    with st.expander("Preview hasil CSV", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Report text", expanded=False):
        if report_text:
            st.code(report_text)
        else:
            st.write("Report belum tersedia.")


page = st.sidebar.radio("Halaman", options=["Dashboard", "History"])

if page == "History":
    render_history_page()
    st.stop()

st.title("Text Clustering Dashboard")
st.caption("Atur parameter, jalankan KMeans/DBSCAN/HDBSCAN, lalu lihat hasil dan grafik cluster dari browser.")

available_inputs = csv_candidates()
default_input = str(DATA_RAW_DIR / "export.csv")
selected_input = st.sidebar.selectbox(
    "Input CSV",
    options=available_inputs if available_inputs else [default_input],
    index=(available_inputs.index(default_input) if default_input in available_inputs else 0),
)
custom_input = st.sidebar.text_input("Atau pakai path custom", value=selected_input)
input_path = custom_input.strip() or selected_input

detected_columns = detect_columns(input_path)
default_text_col = "text" if "text" in detected_columns else (detected_columns[0] if detected_columns else "text")
text_col = st.sidebar.selectbox("Kolom teks", options=detected_columns or ["text"], index=(detected_columns.index(default_text_col) if default_text_col in detected_columns else 0))

st.sidebar.header("Pengaturan Umum")
min_len = st.sidebar.number_input("Min text length", min_value=0, value=10)
min_words = st.sidebar.number_input("Min words", min_value=0, value=5)
limit = st.sidebar.number_input("Limit rows", min_value=0, value=1000)
limit_random = st.sidebar.checkbox("Random sampling saat limit", value=True)
drop_duplicate_texts = st.sidebar.checkbox("Drop duplicate cleaned texts", value=True)
preview_rows = st.sidebar.number_input("Preview rows preprocessing", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, value=64)
sample_size = st.sidebar.number_input("Sample size metric", min_value=2, value=1000)
embedding_model = st.sidebar.text_input(
    "Embedding model",
    value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
device = st.sidebar.text_input("Device", value="")
keep_hashtags = st.sidebar.checkbox("Keep hashtag di embedding", value=False)
remove_emojis = st.sidebar.checkbox("Hapus emoji di embedding", value=False)
stopwords = st.sidebar.text_input("Stopwords path", value=str(RESOURCES_DIR / "stopword.txt"))
no_default_stopwords = st.sidebar.checkbox("Disable default stopwords", value=False)

st.sidebar.header("KMeans")
k = st.sidebar.number_input("Jumlah cluster (k)", min_value=1, value=5)
kmeans_output = st.sidebar.text_input("Output CSV KMeans", value=str(DATA_PROCESSED_DIR / "export_clustered.csv"))
kmeans_report = st.sidebar.text_input("Report KMeans", value=str(REPORTS_DIR / "cluster_report.txt"))
kmeans_extra = st.sidebar.text_input("Extra args KMeans", value="")

st.sidebar.header("DBSCAN")
eps = st.sidebar.number_input("eps", min_value=0.0, value=0.35, step=0.01, format="%.2f")
min_samples = st.sidebar.number_input("min_samples", min_value=1, value=5)
dbscan_use_umap = st.sidebar.checkbox("Use UMAP for DBSCAN (fallback PCA)", value=False)
dbscan_umap_components = st.sidebar.number_input("DBSCAN UMAP components", min_value=2, value=10)
dbscan_umap_neighbors = st.sidebar.number_input("DBSCAN UMAP neighbors", min_value=2, value=15)
dbscan_umap_min_dist = st.sidebar.number_input("DBSCAN UMAP min_dist", min_value=0.0, value=0.0, step=0.01, format="%.2f")
st.sidebar.caption("Kalau paket UMAP belum tersedia di environment, toggle ini otomatis pakai PCA sebagai fallback.")
dbscan_output = st.sidebar.text_input("Output CSV DBSCAN", value=str(DATA_PROCESSED_DIR / "export_dbscan_clustered.csv"))
dbscan_report = st.sidebar.text_input("Report DBSCAN", value=str(REPORTS_DIR / "dbscan_cluster_report.txt"))
dbscan_extra = st.sidebar.text_input("Extra args DBSCAN", value="")

st.sidebar.header("HDBSCAN")
hdbscan_min_cluster_size = st.sidebar.number_input("HDBSCAN min_cluster_size", min_value=2, value=8)
hdbscan_min_samples = st.sidebar.number_input("HDBSCAN min_samples", min_value=1, value=2)
hdbscan_cluster_selection_epsilon = st.sidebar.number_input("HDBSCAN cluster_selection_epsilon", min_value=0.0, value=0.0, step=0.01, format="%.2f")
hdbscan_use_umap = st.sidebar.checkbox("Use UMAP for HDBSCAN (fallback PCA)", value=False)
hdbscan_umap_components = st.sidebar.number_input("HDBSCAN UMAP components", min_value=2, value=10)
hdbscan_umap_neighbors = st.sidebar.number_input("HDBSCAN UMAP neighbors", min_value=2, value=15)
hdbscan_umap_min_dist = st.sidebar.number_input("HDBSCAN UMAP min_dist", min_value=0.0, value=0.0, step=0.01, format="%.2f")
st.sidebar.caption("HDBSCAN juga akan otomatis pakai PCA kalau UMAP belum terinstall.")
hdbscan_output = st.sidebar.text_input("Output CSV HDBSCAN", value=str(DATA_PROCESSED_DIR / "export_hdbscan_clustered.csv"))
hdbscan_report = st.sidebar.text_input("Report HDBSCAN", value=str(REPORTS_DIR / "hdbscan_cluster_report.txt"))
hdbscan_extra = st.sidebar.text_input("Extra args HDBSCAN", value="")

config = {
    "input": input_path,
    "text_col": text_col,
    "min_len": int(min_len),
    "min_words": int(min_words),
    "limit": int(limit),
    "limit_random": limit_random,
    "drop_duplicate_texts": drop_duplicate_texts,
    "batch_size": int(batch_size),
    "sample_size": int(sample_size),
    "embedding_model": embedding_model,
    "device": device,
    "keep_hashtags": keep_hashtags,
    "remove_emojis": remove_emojis,
    "stopwords": stopwords,
    "no_default_stopwords": no_default_stopwords,
    "k": int(k),
    "kmeans_output": kmeans_output,
    "kmeans_report": kmeans_report,
    "kmeans_extra": kmeans_extra,
    "eps": float(eps),
    "min_samples": int(min_samples),
    "dbscan_use_umap": dbscan_use_umap,
    "dbscan_umap_components": int(dbscan_umap_components),
    "dbscan_umap_neighbors": int(dbscan_umap_neighbors),
    "dbscan_umap_min_dist": float(dbscan_umap_min_dist),
    "dbscan_output": dbscan_output,
    "dbscan_report": dbscan_report,
    "dbscan_extra": dbscan_extra,
    "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
    "hdbscan_min_samples": int(hdbscan_min_samples),
    "hdbscan_cluster_selection_epsilon": float(hdbscan_cluster_selection_epsilon),
    "hdbscan_use_umap": hdbscan_use_umap,
    "hdbscan_umap_components": int(hdbscan_umap_components),
    "hdbscan_umap_neighbors": int(hdbscan_umap_neighbors),
    "hdbscan_umap_min_dist": float(hdbscan_umap_min_dist),
    "hdbscan_output": hdbscan_output,
    "hdbscan_report": hdbscan_report,
    "hdbscan_extra": hdbscan_extra,
}

python_bin = sys.executable
kmeans_command = build_kmeans_command(python_bin, config)
dbscan_command = build_dbscan_command(python_bin, config)
hdbscan_command = build_hdbscan_command(python_bin, config)
history_commands = {
    "KMeans": kmeans_command,
    "DBSCAN": dbscan_command,
    "HDBSCAN": hdbscan_command,
}

st.subheader("Run")
button_col1, button_col2, button_col3, button_col4 = st.columns(4)
run_kmeans = button_col1.button("Run KMeans", use_container_width=True)
run_dbscan = button_col2.button("Run DBSCAN", use_container_width=True)
run_hdbscan = button_col3.button("Run HDBSCAN", use_container_width=True)
run_all = button_col4.button("Run All", use_container_width=True)

log_placeholder = st.empty()

if run_kmeans or run_dbscan or run_hdbscan or run_all:
    commands: list[tuple[str, list[str]]] = []
    if run_kmeans:
        commands.append(("KMeans", kmeans_command))
    if run_dbscan:
        commands.append(("DBSCAN", dbscan_command))
    if run_hdbscan:
        commands.append(("HDBSCAN", hdbscan_command))
    if run_all:
        commands.extend([("KMeans", kmeans_command), ("DBSCAN", dbscan_command), ("HDBSCAN", hdbscan_command)])

    output_blocks: list[str] = []
    had_error = False
    for label, command in commands:
        with st.spinner(f"Menjalankan {label}..."):
            code, output = run_command(command)
        output_blocks.append(f"$ {' '.join(command)}\n\n{output}")
        if code != 0:
            had_error = True
            break
        append_history_record(build_history_record(label, history_commands[label], config))

    if had_error:
        log_placeholder.error("Ada proses yang gagal. Cek log di bawah.")
    else:
        log_placeholder.success("Proses selesai.")

    with st.expander("Log eksekusi", expanded=True):
        for block in output_blocks:
            st.code(block)

st.subheader("Command Preview")
preview_tab1, preview_tab2, preview_tab3 = st.tabs(["KMeans", "DBSCAN", "HDBSCAN"])
with preview_tab1:
    st.code(" ".join(kmeans_command))
with preview_tab2:
    st.code(" ".join(dbscan_command))
with preview_tab3:
    st.code(" ".join(hdbscan_command))

st.subheader("Preview Preprocessing")
st.caption("Panel ini menampilkan teks asli, preprocessing umum untuk filtering/report, dan hasil tiap tahap preprocessing embedding yang menjaga makna pesan.")
preprocess_preview = build_preprocessing_preview(
    input_path,
    text_col,
    rows=int(preview_rows),
    remove_hashtags=(not keep_hashtags),
    remove_emojis=remove_emojis,
)
if preprocess_preview is None:
    st.info("Preview preprocessing belum tersedia. Pastikan file input dan kolom teks valid.")
else:
    st.dataframe(preprocess_preview, use_container_width=True)

result_tab1, result_tab2, result_tab3 = st.tabs(["Hasil KMeans", "Hasil DBSCAN", "Hasil HDBSCAN"])
with result_tab1:
    render_result("KMeans", kmeans_output, kmeans_report)
with result_tab2:
    render_result("DBSCAN", dbscan_output, dbscan_report)
with result_tab3:
    render_result("HDBSCAN", hdbscan_output, hdbscan_report)
