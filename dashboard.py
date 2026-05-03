import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cluster_commands import build_dbscan_command, build_kmeans_command
from project_paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR, RESOURCES_DIR


st.set_page_config(page_title="Text Clustering Dashboard", layout="wide")


def csv_candidates() -> list[str]:
    patterns = [DATA_RAW_DIR.glob("*.csv"), (PROJECT_ROOT / "data" / "sample").glob("*.csv")]
    files: list[str] = []
    for pattern in patterns:
        files.extend(str(path) for path in sorted(pattern))
    return files


def detect_columns(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception:
        return []
    return list(df.columns)


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


def cluster_chart_data(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or "cluster" not in df.columns:
        return pd.DataFrame(columns=["cluster", "count"])
    counts = df["cluster"].value_counts(dropna=False).sort_index()
    chart_df = counts.rename_axis("cluster").reset_index(name="count")
    chart_df["cluster"] = chart_df["cluster"].astype(str)
    return chart_df


def render_result(title: str, output_path: str, report_path: str) -> None:
    st.subheader(title)
    df, summary = summarize_result(output_path)
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

    chart_df = cluster_chart_data(df)
    if not chart_df.empty:
        st.bar_chart(chart_df.set_index("cluster"))

    with st.expander("Preview hasil CSV", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    report_text = load_report_text(report_path)
    with st.expander("Report text", expanded=False):
        if report_text:
            st.code(report_text)
        else:
            st.write("Report belum tersedia.")


st.title("Text Clustering Dashboard")
st.caption("Atur parameter, jalankan KMeans/DBSCAN, lalu lihat hasil dan grafik cluster dari browser.")

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
batch_size = st.sidebar.number_input("Batch size", min_value=1, value=64)
sample_size = st.sidebar.number_input("Sample size metric", min_value=2, value=1000)
embedding_model = st.sidebar.text_input(
    "Embedding model",
    value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
device = st.sidebar.text_input("Device", value="")
stopwords = st.sidebar.text_input("Stopwords path", value=str(RESOURCES_DIR / "stopword.txt"))
no_default_stopwords = st.sidebar.checkbox("Disable default stopwords", value=False)

st.sidebar.header("KMeans")
k = st.sidebar.number_input("Jumlah cluster (k)", min_value=1, value=10)
kmeans_output = st.sidebar.text_input("Output CSV KMeans", value=str(DATA_PROCESSED_DIR / "export_clustered.csv"))
kmeans_report = st.sidebar.text_input("Report KMeans", value=str(REPORTS_DIR / "cluster_report.txt"))
kmeans_extra = st.sidebar.text_input("Extra args KMeans", value="")

st.sidebar.header("DBSCAN")
eps = st.sidebar.number_input("eps", min_value=0.0, value=0.22, step=0.01, format="%.2f")
min_samples = st.sidebar.number_input("min_samples", min_value=1, value=5)
dbscan_output = st.sidebar.text_input("Output CSV DBSCAN", value=str(DATA_PROCESSED_DIR / "export_dbscan_clustered.csv"))
dbscan_report = st.sidebar.text_input("Report DBSCAN", value=str(REPORTS_DIR / "dbscan_cluster_report.txt"))
dbscan_extra = st.sidebar.text_input("Extra args DBSCAN", value="")

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
    "stopwords": stopwords,
    "no_default_stopwords": no_default_stopwords,
    "k": int(k),
    "kmeans_output": kmeans_output,
    "kmeans_report": kmeans_report,
    "kmeans_extra": kmeans_extra,
    "eps": float(eps),
    "min_samples": int(min_samples),
    "dbscan_output": dbscan_output,
    "dbscan_report": dbscan_report,
    "dbscan_extra": dbscan_extra,
}

python_bin = sys.executable
kmeans_command = build_kmeans_command(python_bin, config)
dbscan_command = build_dbscan_command(python_bin, config)

st.subheader("Run")
button_col1, button_col2, button_col3 = st.columns(3)
run_kmeans = button_col1.button("Run KMeans", use_container_width=True)
run_dbscan = button_col2.button("Run DBSCAN", use_container_width=True)
run_both = button_col3.button("Run Both", use_container_width=True)

log_placeholder = st.empty()

if run_kmeans or run_dbscan or run_both:
    commands: list[tuple[str, list[str]]] = []
    if run_kmeans:
        commands.append(("KMeans", kmeans_command))
    if run_dbscan:
        commands.append(("DBSCAN", dbscan_command))
    if run_both:
        commands.extend([("KMeans", kmeans_command), ("DBSCAN", dbscan_command)])

    output_blocks: list[str] = []
    had_error = False
    for label, command in commands:
        with st.spinner(f"Menjalankan {label}..."):
            code, output = run_command(command)
        output_blocks.append(f"$ {' '.join(command)}\n\n{output}")
        if code != 0:
            had_error = True
            break

    if had_error:
        log_placeholder.error("Ada proses yang gagal. Cek log di bawah.")
    else:
        log_placeholder.success("Proses selesai.")

    with st.expander("Log eksekusi", expanded=True):
        for block in output_blocks:
            st.code(block)

st.subheader("Command Preview")
preview_tab1, preview_tab2 = st.tabs(["KMeans", "DBSCAN"])
with preview_tab1:
    st.code(" ".join(kmeans_command))
with preview_tab2:
    st.code(" ".join(dbscan_command))

result_tab1, result_tab2 = st.tabs(["Hasil KMeans", "Hasil DBSCAN"])
with result_tab1:
    render_result("KMeans", kmeans_output, kmeans_report)
with result_tab2:
    render_result("DBSCAN", dbscan_output, dbscan_report)
