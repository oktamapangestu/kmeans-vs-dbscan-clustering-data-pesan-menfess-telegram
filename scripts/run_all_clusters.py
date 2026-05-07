import argparse
import shlex
import subprocess
import sys

from cluster_commands import build_dbscan_command, build_hdbscan_command, build_kmeans_command
from project_paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR


def run_command(label: str, command: list[str], *, dry_run: bool) -> None:
    rendered = shlex.join(command)
    print(f"[{label}] {rendered}")
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run both KMeans and DBSCAN clustering in one command")
    ap.add_argument("--input", default=str(DATA_RAW_DIR / "export.csv"), help="Input CSV path")
    ap.add_argument("--text-col", default="text", help="Text column name")
    ap.add_argument("--min-len", type=int, default=10, help="Drop rows with cleaned text shorter than this")
    ap.add_argument("--min-words", type=int, default=5, help="Drop rows with cleaned text having fewer words than this")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only use first N rows after filtering")
    ap.add_argument("--limit-random", action="store_true", help="When used with --limit, sample rows randomly after filtering")
    ap.add_argument("--drop-duplicate-texts", action="store_true", help="Drop duplicate cleaned texts and keep the first occurrence only")
    ap.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Sentence-BERT-compatible model name supported by fastembed or sentence-transformers")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--device", default="", help="Device for sentence-transformers, e.g. cpu, mps, cuda")
    ap.add_argument("--sample-size", type=int, default=5000, help="Silhouette sample size")
    ap.add_argument("--stopwords", default="", help="Path to stopwords file (one word per line)")
    ap.add_argument("--no-default-stopwords", action="store_true", help="Disable default stopwords from resources/stopword.txt")

    ap.add_argument("--k", type=int, default=10, help="Number of clusters for KMeans")
    ap.add_argument("--kmeans-output", default=str(DATA_PROCESSED_DIR / "export_clustered.csv"), help="Output CSV path for KMeans")
    ap.add_argument("--kmeans-report", default=str(REPORTS_DIR / "cluster_report.txt"), help="Report path for KMeans")
    ap.add_argument("--kmeans-extra", default="", help="Extra raw arguments appended to kmeans_text_cluster.py")

    ap.add_argument("--eps", type=float, default=0.85, help="DBSCAN eps (cosine distance)")
    ap.add_argument("--min-samples", type=int, default=10, help="DBSCAN min_samples")
    ap.add_argument("--dbscan-output", default=str(DATA_PROCESSED_DIR / "export_dbscan_clustered.csv"), help="Output CSV path for DBSCAN")
    ap.add_argument("--dbscan-report", default=str(REPORTS_DIR / "dbscan_cluster_report.txt"), help="Report path for DBSCAN")
    ap.add_argument("--dbscan-extra", default="", help="Extra raw arguments appended to dbscan_text_cluster.py")

    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=15, help="Minimum cluster size for HDBSCAN")
    ap.add_argument("--hdbscan-min-samples", type=int, default=5, help="min_samples for HDBSCAN")
    ap.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.0, help="cluster_selection_epsilon for HDBSCAN")
    ap.add_argument("--hdbscan-output", default=str(DATA_PROCESSED_DIR / "export_hdbscan_clustered.csv"), help="Output CSV path for HDBSCAN")
    ap.add_argument("--hdbscan-report", default=str(REPORTS_DIR / "hdbscan_cluster_report.txt"), help="Report path for HDBSCAN")
    ap.add_argument("--hdbscan-extra", default="", help="Extra raw arguments appended to hdbscan_text_cluster.py")

    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = ap.parse_args()

    python_bin = sys.executable
    config = vars(args)

    kmeans_cmd = build_kmeans_command(python_bin, config)
    dbscan_cmd = build_dbscan_command(python_bin, config)
    hdbscan_cmd = build_hdbscan_command(python_bin, config)

    run_command("KMeans", kmeans_cmd, dry_run=args.dry_run)
    run_command("DBSCAN", dbscan_cmd, dry_run=args.dry_run)
    run_command("HDBSCAN", hdbscan_cmd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
