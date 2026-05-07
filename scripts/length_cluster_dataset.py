import argparse
import re

import pandas as pd
from sklearn.cluster import KMeans

from project_paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, ensure_dir, ensure_parent_dir


LENGTH_LABELS = [
    "pendek",
    "menengah_pendek",
    "menengah",
    "panjang",
    "sangat_panjang",
]


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def average_word_length(text: str) -> float:
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster raw dataset by text length")
    ap.add_argument("--input", default=str(DATA_RAW_DIR / "export.csv"), help="Input CSV")
    ap.add_argument(
        "--output-csv",
        default=str(DATA_PROCESSED_DIR / "export_length_clustered.csv"),
        help="Combined output CSV",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DATA_PROCESSED_DIR / "export_length_clusters"),
        help="Directory for per-cluster CSV files",
    )
    ap.add_argument("--text-col", default="text", help="Text column name")
    ap.add_argument("--k", type=int, default=5, help="Number of length clusters")
    ap.add_argument("--random-state", type=int, default=42, help="Random state for KMeans")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    texts = df[args.text_col].fillna("").astype(str)
    df2 = df.copy()
    df2["word_count"] = texts.map(count_words)
    df2["char_count"] = texts.str.replace(r"\s+", " ", regex=True).str.strip().str.len()
    df2["avg_word_length"] = texts.map(average_word_length).round(3)

    if len(df2) < args.k:
        raise SystemExit(f"Not enough rows ({len(df2)}) for k={args.k}")

    km = KMeans(n_clusters=args.k, n_init="auto", random_state=args.random_state)
    raw_labels = km.fit_predict(df2[["word_count"]])

    centers = pd.Series(km.cluster_centers_.ravel())
    ordered_cluster_ids = centers.sort_values().index.tolist()
    remap = {old_id: new_id for new_id, old_id in enumerate(ordered_cluster_ids)}
    df2["length_cluster"] = pd.Series(raw_labels).map(remap).astype(int)

    label_map = {idx: LENGTH_LABELS[idx] if idx < len(LENGTH_LABELS) else f"cluster_{idx}" for idx in range(args.k)}
    center_map = {remap[old_id]: round(float(center), 3) for old_id, center in centers.items()}
    df2["length_cluster_label"] = df2["length_cluster"].map(label_map)
    df2["length_cluster_center_words"] = df2["length_cluster"].map(center_map)

    ensure_parent_dir(args.output_csv)
    df2.to_csv(args.output_csv, index=False)

    ensure_dir(args.output_dir)
    for cluster_id in sorted(df2["length_cluster"].unique().tolist()):
        cluster_df = df2[df2["length_cluster"] == cluster_id].copy()
        cluster_label = label_map[cluster_id]
        out_path = f"{args.output_dir}/cluster_{cluster_id}_{cluster_label}.csv"
        cluster_df.to_csv(out_path, index=False)

    counts = df2["length_cluster"].value_counts().sort_index()
    print(f"Wrote combined CSV: {args.output_csv}")
    print(f"Wrote cluster directory: {args.output_dir}")
    for cluster_id, count in counts.items():
        print(
            f"cluster={cluster_id} label={label_map[int(cluster_id)]} center_words={center_map[int(cluster_id)]} rows={int(count)}"
        )


if __name__ == "__main__":
    main()
