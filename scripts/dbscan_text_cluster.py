import argparse
from functools import partial

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA

# Reuse the same cleaning, embedding, and keyword report logic.
from kmeans_text_cluster import (
    clean_text,
    clean_text_for_embedding,
    encode_texts,
    load_embedding_model,
    load_stopwords,
    remove_stopwords_from_texts,
    top_terms_per_cluster,
)
from project_paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR, ensure_parent_dir
from text_dataset import prepare_text_dataset

try:
    from umap import UMAP
except ImportError:
    UMAP = None


def exemplar_texts(X: np.ndarray, labels: np.ndarray, texts: list[str], per_cluster: int):
    """Pick exemplar texts nearest each cluster centroid in embedding space."""
    out: dict[int, list[str]] = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            out[int(c)] = []
            continue

        centroid = X[idx].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = X[idx] @ centroid
        best_local = np.argsort(-sims)[:per_cluster]
        best = idx[best_local]
        out[int(c)] = [texts[i] for i in best]
    return out


def compute_dbscan_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> tuple[float | None, float | None, str | None]:
    mask = labels != -1
    non_noise_labels = labels[mask]

    if mask.sum() >= 2 and len(set(non_noise_labels.tolist())) >= 2:
        metric_X = X[mask]
        metric_labels = non_noise_labels
        metric_scope = "non_noise"
    elif len(set(labels.tolist())) >= 2:
        metric_X = X
        metric_labels = labels
        metric_scope = "all_labels"
    else:
        return None, None, None

    sil = None
    metric_sample = min(sample_size, int(len(metric_labels)))
    if metric_sample >= 2:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(metric_labels), size=metric_sample, replace=False)
        sampled_labels = metric_labels[sample_idx]
        if len(set(sampled_labels.tolist())) >= 2:
            sil = float(silhouette_score(metric_X[sample_idx], sampled_labels, metric="cosine"))

    dbi = float(davies_bouldin_score(metric_X, metric_labels)) if len(set(metric_labels.tolist())) >= 2 else None
    return sil, dbi, metric_scope


def maybe_apply_umap(
    X: np.ndarray,
    *,
    enabled: bool,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[np.ndarray, str]:
    if not enabled:
        return X, "none"
    if UMAP is None:
        reduced = PCA(n_components=min(n_components, X.shape[1]), random_state=random_state).fit_transform(X)
        return np.asarray(reduced, dtype=np.float32), "pca_fallback"
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(X), dtype=np.float32), "umap"


def main():
    ap = argparse.ArgumentParser(description="DBSCAN clustering for data/raw/export.csv text column using Sentence-BERT")
    ap.add_argument("--input", default=str(DATA_RAW_DIR / "export.csv"), help="Input CSV path")
    ap.add_argument("--text-col", default="text", help="Text column name")
    ap.add_argument("--min-len", type=int, default=10, help="Drop rows with cleaned text shorter than this")
    ap.add_argument("--min-words", type=int, default=5, help="Drop rows with cleaned text having fewer words than this")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only use first N rows after filtering")
    ap.add_argument("--limit-random", action="store_true", help="When used with --limit, sample rows randomly after filtering")
    ap.add_argument(
        "--drop-duplicate-texts",
        action="store_true",
        help="Drop duplicate cleaned texts and keep the first occurrence only",
    )
    ap.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Sentence-BERT-compatible model name supported by fastembed or sentence-transformers")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--device", default="", help="Device for sentence-transformers, e.g. cpu, mps, cuda")
    ap.add_argument("--keep-hashtags", action="store_true", help="Keep hashtags in embedding preprocessing")
    ap.add_argument("--remove-emojis", action="store_true", help="Remove emojis in embedding preprocessing")
    ap.add_argument("--sample-size", type=int, default=5000, help="Silhouette sample size")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")

    ap.add_argument("--max-features", type=int, default=10000, help="TF-IDF max features for report top terms")
    ap.add_argument("--min-df", type=int, default=5, help="TF-IDF min document frequency for report top terms")
    ap.add_argument("--max-df", type=float, default=1.0, help="TF-IDF max document frequency for report top terms")
    ap.add_argument("--ngram-max", type=int, default=2, help="Max ngram length for report top terms")
    ap.add_argument("--stopwords", default="", help="Path to stopwords file (one word per line)")
    ap.add_argument("--no-default-stopwords", action="store_true", help="Disable default stopwords (resources/stopword.txt)")

    ap.add_argument("--eps", type=float, default=0.85, help="DBSCAN eps (cosine distance)")
    ap.add_argument("--min-samples", type=int, default=10, help="DBSCAN min_samples")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (DBSCAN)")
    ap.add_argument("--use-umap", action="store_true", help="Reduce embeddings with UMAP before DBSCAN")
    ap.add_argument("--umap-components", type=int, default=10, help="UMAP output dimensions")
    ap.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    ap.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist")
    ap.add_argument(
        "--eps-scan",
        default="",
        help="Comma-separated eps values to scan (runs on current data and exits)",
    )

    ap.add_argument("--top-terms", type=int, default=15, help="Top terms per cluster")
    ap.add_argument("--examples", type=int, default=5, help="Example texts per cluster")
    ap.add_argument("--output-csv", default=str(DATA_PROCESSED_DIR / "export_dbscan_clustered.csv"), help="Output CSV with cluster labels")
    ap.add_argument("--report", default=str(REPORTS_DIR / "dbscan_cluster_report.txt"), help="Output report path")
    ap.add_argument("--clean-col", default="text_clean", help="Name of the cleaned text column to write")
    args = ap.parse_args()

    prepared = prepare_text_dataset(
        input_path=args.input,
        text_col=args.text_col,
        min_len=args.min_len,
        min_words=args.min_words,
        limit=args.limit,
        limit_random=args.limit_random,
        drop_duplicate_texts=args.drop_duplicate_texts,
        random_state=args.random_state,
        clean_col=args.clean_col,
        clean_text=clean_text,
        clean_text_embed=partial(
            clean_text_for_embedding,
            remove_hashtags=(not args.keep_hashtags),
            remove_emojis=args.remove_emojis,
        ),
    )
    df2 = prepared.df
    cleaned2 = prepared.cleaned_texts
    cleaned_embed = prepared.cleaned_texts_embed
    dedup_removed = prepared.dedup_removed

    stop_set = load_stopwords(
        args.stopwords if args.stopwords else None,
        use_default=(not args.no_default_stopwords),
    ) or set()
    stop = sorted(stop_set) if stop_set else None

    model = load_embedding_model(args.embedding_model, device=args.device)
    X = encode_texts(model, cleaned_embed, batch_size=args.batch_size)
    X_cluster, reducer_used = maybe_apply_umap(
        X,
        enabled=args.use_umap,
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )

    if args.eps_scan.strip():
        eps_vals = [float(x.strip()) for x in args.eps_scan.split(",") if x.strip()]
        print(f"eps_scan on n={X_cluster.shape[0]} ...")
        for eps in eps_vals:
            db = DBSCAN(
                eps=eps,
                min_samples=args.min_samples,
                metric=("euclidean" if reducer_used != "none" else "cosine"),
                n_jobs=args.n_jobs,
            )
            lab = db.fit_predict(X_cluster)
            unique, counts = np.unique(lab, return_counts=True)
            count_map = dict(zip(unique.tolist(), counts.tolist()))
            n_noise = int(count_map.get(-1, 0))
            n_clusters = len([c for c in unique.tolist() if c != -1])
            sil, dbi, metric_scope = compute_dbscan_metrics(
                X_cluster,
                lab,
                sample_size=args.sample_size,
                random_state=args.random_state,
            )
            print(
                f"eps={eps:.3f} clusters={n_clusters} noise_rate={n_noise/max(1,len(lab)):.3f} metric_scope={metric_scope or 'NA'} silhouette={('NA' if sil is None else f'{sil:.4f}')} davies_bouldin={('NA' if dbi is None else f'{dbi:.4f}')}"
            )
        return

    db = DBSCAN(
        eps=args.eps,
        min_samples=args.min_samples,
        metric=("euclidean" if reducer_used != "none" else "cosine"),
        n_jobs=args.n_jobs,
    )
    labels = db.fit_predict(X_cluster)

    df2["cluster"] = labels
    ensure_parent_dir(args.output_csv)
    df2.to_csv(args.output_csv, index=False)

    # Basic report.
    unique, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    n_noise = int(count_map.get(-1, 0))
    n_clusters = len([c for c in unique.tolist() if c != -1])

    sil, dbi, metric_scope = compute_dbscan_metrics(
        X_cluster,
        labels,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    keyword_texts = remove_stopwords_from_texts(cleaned2, stop_set)
    tops = top_terms_per_cluster(
        keyword_texts,
        labels,
        stop_words=stop,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        top_n=args.top_terms,
    )
    ex = exemplar_texts(X, labels, cleaned2, per_cluster=args.examples)

    ensure_parent_dir(args.report)
    with open(args.report, "w", encoding="utf-8") as f:
        if args.limit and args.limit > 0:
            limit_mode = "random" if args.limit_random else "head"
            limit_note = f", limit={args.limit}, limit_mode={limit_mode}"
        else:
            limit_note = ""
        dedup_note = f", dedup_removed={dedup_removed}" if args.drop_duplicate_texts else ""
        f.write(f"Rows used: {len(df2)} / {prepared.total_rows} (min_len={args.min_len}, min_words={args.min_words}{limit_note}{dedup_note})\n")
        f.write(f"embedding_model={args.embedding_model}\n")
        cluster_metric = "euclidean" if reducer_used != "none" else "cosine"
        umap_note = (
            f" use_umap=True reducer={reducer_used} umap_components={args.umap_components} umap_neighbors={args.umap_neighbors} umap_min_dist={args.umap_min_dist}"
            if args.use_umap
            else " use_umap=False reducer=none"
        )
        f.write(f"eps={args.eps} min_samples={args.min_samples} metric={cluster_metric}{umap_note}\n")
        f.write(f"clusters={n_clusters} noise_points={n_noise} noise_rate={n_noise/max(1,len(df2)):.4f}\n")
        if sil is not None:
            sample_rows = min(args.sample_size, len(labels) if metric_scope == "all_labels" else int((labels != -1).sum()))
            f.write(f"silhouette_cosine({metric_scope}_sample={sample_rows})={sil:.4f}\n")
        else:
            f.write("silhouette_cosine=NA\n")
        f.write(f"davies_bouldin({metric_scope or 'NA'})={('NA' if dbi is None else f'{dbi:.4f}')}\n")
        f.write("\n")

        cluster_sizes = [(c, count_map[c]) for c in unique.tolist() if c != -1]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        for c, n in cluster_sizes:
            f.write(f"=== Cluster {c} (n={n}) ===\n")
            f.write("Top terms: " + ", ".join(tops.get(c, [])) + "\n")
            for i, t in enumerate(ex.get(c, []), start=1):
                t = t.replace("\r", " ").replace("\n", " ").strip()
                if len(t) > 280:
                    t = t[:277] + "..."
                f.write(f"{i}. {t}\n")
            f.write("\n")

        if n_noise:
            f.write("=== Noise (-1) sample ===\n")
            noise_idx = np.where(labels == -1)[0][: min(10, n_noise)]
            for i, j in enumerate(noise_idx, start=1):
                t = cleaned2[j].replace("\r", " ").replace("\n", " ").strip()
                if len(t) > 280:
                    t = t[:277] + "..."
                f.write(f"{i}. {t}\n")

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
