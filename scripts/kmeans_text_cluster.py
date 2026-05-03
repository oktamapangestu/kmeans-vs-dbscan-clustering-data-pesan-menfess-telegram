import argparse
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from project_paths import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    REPORTS_DIR,
    RESOURCES_DIR,
    ensure_parent_dir,
)
from text_dataset import prepare_text_dataset


_URL_RE = re.compile(r"https?://\S+|www\.[^\s]+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#[-\w]+")
_NON_WORD_RE = re.compile(r"[^0-9a-zA-Z\s]+")
_MULTISPACE_RE = re.compile(r"\s+")
_ELONGATED_END_RE = re.compile(r"([a-zA-Z])\1+$")

_BOILERPLATE_PATTERNS = [
    re.compile(r"\bsansfess\b"),
    re.compile(r"\bkirimin\s+aku\s+pesan\s+rahasia\b"),
    re.compile(r"\bpesan\s+rahasia\b"),
]

DEFAULT_STOPWORDS_FILE = RESOURCES_DIR / "stopword.txt"

_EMBEDDING_MODEL_ALIASES = {
    "intfloat/multilingual-e5-base": "intfloat/multilingual-e5-large",
}

_TOKEN_NORMALIZATION_MAP = {
    "aja": "saja",
    "aj": "saja",
    "bgt": "banget",
    "bgtt": "banget",
    "blm": "belum",
    "bs": "bisa",
    "bsa": "bisa",
    "dgn": "dengan",
    "dr": "dari",
    "g": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "gx": "tidak",
    "jg": "juga",
    "jd": "jadi",
    "krn": "karena",
    "lg": "lagi",
    "org": "orang",
    "sdh": "sudah",
    "sm": "sama",
    "sy": "saya",
    "tdk": "tidak",
    "tp": "tapi",
    "utk": "untuk",
    "yg": "yang",
    "ngga": "tidak",
    "nggak": "tidak",
    "nggaak": "tidak",
    "nggk": "tidak",
}


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)

    s = s.lower()
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(" ", s)
    for pat in _BOILERPLATE_PATTERNS:
        s = pat.sub(" ", s)
    s = _NON_WORD_RE.sub(" ", s)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    s = normalize_elongated_words(s)
    s = normalize_tokens(s)
    return s


def normalize_elongated_words(text: str) -> str:
    if not text:
        return text

    words: list[str] = []
    for word in text.split():
        # Normalize informal emphasis at the end of a word: akuu -> aku, iniii -> ini.
        words.append(_ELONGATED_END_RE.sub(r"\1", word))
    return " ".join(words)


def normalize_tokens(text: str) -> str:
    if not text:
        return text

    return " ".join(_TOKEN_NORMALIZATION_MAP.get(word, word) for word in text.split())


def load_stopwords(path: str | None, *, use_default: bool) -> set[str] | None:
    out: set[str] = set()
    if use_default:
        try:
            with open(DEFAULT_STOPWORDS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        out.add(s)
        except FileNotFoundError:
            pass

    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    out.add(s)

    return out or None
def auto_stopwords_from_low_idf(
    texts: list[str],
    *,
    base_stopwords: set[str],
    top_n: int,
    min_df_ratio: float,
    max_features: int,
) -> set[str]:
    if top_n <= 0 or not texts:
        return set()

    probe = TfidfVectorizer(
        max_features=max_features if max_features > 0 else None,
        min_df=max(2, int(0.001 * max(1, len(texts)))),
        ngram_range=(1, 1),
        stop_words=sorted(base_stopwords) if base_stopwords else None,
        norm=None,
    )
    X = probe.fit_transform(texts)
    terms = np.array(probe.get_feature_names_out())
    idf = np.array(probe.idf_)
    df = np.asarray((X > 0).sum(axis=0)).reshape(-1)
    df_ratio = df / max(1, X.shape[0])
    eligible = np.where(df_ratio >= float(min_df_ratio))[0] if min_df_ratio > 0 else np.arange(len(terms))
    if eligible.size == 0:
        eligible = np.arange(len(terms))
    order = eligible[np.argsort(idf[eligible], kind="mergesort")]
    picked = order[: min(top_n, order.size)]
    return set(terms[picked].tolist())


def remove_stopwords_from_texts(texts: list[str], stopwords: set[str]) -> list[str]:
    if not stopwords:
        return texts
    out: list[str] = []
    for text in texts:
        words = [w for w in text.split() if w not in stopwords]
        out.append(" ".join(words))
    return out


@dataclass(frozen=True)
class EvalResult:
    inertia: float
    silhouette_cosine: float | None
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari_mean: float | None
    stability_ari_std: float | None


def fit_kmeans_model(X: np.ndarray, *, k: int, random_state: int):
    actual_k = max(1, min(k, X.shape[0]))
    kmeans_params = dict(n_clusters=actual_k, n_init="auto")
    km = KMeans(random_state=random_state, **kmeans_params)
    labels = km.fit_predict(X)
    return km, labels, kmeans_params, actual_k


def evaluate(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    inertia: float,
    sample_size: int,
    random_state: int,
    stability_seeds: list[int] | None,
    kmeans_params: dict,
) -> EvalResult:
    dbi = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    sil = None
    n = X.shape[0]
    if n >= 2 and len(np.unique(labels)) >= 2:
        ssz = min(sample_size, n)
        if ssz >= 2:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=ssz, replace=False)
            sil = float(silhouette_score(X[idx], labels[idx], metric="cosine"))

    ari_mean = ari_std = None
    if stability_seeds:
        base = labels
        aris: list[float] = []
        for seed in stability_seeds:
            km = KMeans(random_state=seed, **kmeans_params)
            lab = km.fit_predict(X)
            aris.append(float(adjusted_rand_score(base, lab)))
        if aris:
            ari_mean = float(np.mean(aris))
            ari_std = float(np.std(aris))

    return EvalResult(
        inertia=float(inertia),
        silhouette_cosine=sil,
        davies_bouldin=dbi,
        calinski_harabasz=ch,
        stability_ari_mean=ari_mean,
        stability_ari_std=ari_std,
    )


def top_terms_per_cluster(texts: list[str], labels: np.ndarray, *, stop_words: list[str] | None, max_features: int, min_df: int, max_df: float, ngram_max: int, top_n: int) -> dict[int, list[str]]:
    if not texts:
        return {}
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, max(1, ngram_max)),
        stop_words=stop_words,
        norm="l2",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return {int(k): [] for k in np.unique(labels)}
    terms = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for k in sorted(np.unique(labels)):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            out[int(k)] = []
            continue
        centroid = np.asarray(X[idx].mean(axis=0)).reshape(-1)
        top_idx = np.argsort(centroid)[::-1][:top_n]
        out[int(k)] = terms[top_idx].tolist()
    return out


def exemplar_texts(X: np.ndarray, labels: np.ndarray, km: KMeans, texts: list[str], per_cluster: int) -> dict[int, list[str]]:
    centers = km.cluster_centers_
    center_norms = np.linalg.norm(centers, axis=1)
    x_norms = np.linalg.norm(X, axis=1)
    out: dict[int, list[str]] = {}
    for k in range(km.n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            out[k] = []
            continue
        denom = x_norms[idx] * max(center_norms[k], 1e-12)
        sims = (X[idx] @ centers[k]) / np.maximum(denom, 1e-12)
        best = idx[np.argsort(-sims)[:per_cluster]]
        out[k] = [texts[i] for i in best]
    return out


def load_embedding_model(name: str, *, device: str):
    resolved_name = _EMBEDDING_MODEL_ALIASES.get(name, name)
    if resolved_name != name:
        warnings.warn(
            f"Model '{name}' is not supported by fastembed in this environment; using '{resolved_name}' instead.",
            UserWarning,
            stacklevel=2,
        )

    if not _supports_fastembed_model(resolved_name):
        return load_sentence_transformer_model(resolved_name, device=device)

    cuda = device if device else "auto"
    try:
        return TextEmbedding(model_name=resolved_name, cuda=cuda)
    except Exception as exc:
        if _is_unavailable_cuda_provider(exc):
            return TextEmbedding(model_name=resolved_name, cuda=False)

        if not _is_missing_fastembed_model_file(exc):
            raise

        cleared = clear_fastembed_model_cache(resolved_name)
        if not cleared:
            raise RuntimeError(
                f"Failed to load embedding model '{resolved_name}': missing ONNX model file in fastembed cache. "
                "Delete the corrupted fastembed cache and try again."
            ) from exc

        return TextEmbedding(model_name=resolved_name, cuda=cuda)


def _supports_fastembed_model(name: str) -> bool:
    return any(model.get("model", "").lower() == name.lower() for model in TextEmbedding.list_supported_models())


def load_sentence_transformer_model(name: str, *, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"Model '{name}' requires the 'sentence-transformers' package. Install it with: pip install sentence-transformers"
        ) from exc

    target_device = device or None
    return SentenceTransformer(name, device=target_device)


def _is_missing_fastembed_model_file(exc: Exception) -> bool:
    msg = str(exc)
    missing_markers = (
        "NO_SUCHFILE",
        "File doesn't exist",
        "No such file or directory",
        "filesystem error: in file_size",
    )
    model_markers = (
        ".onnx",
        ".onnx_data",
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
    )
    return any(marker in msg for marker in missing_markers) and any(marker in msg for marker in model_markers)


def _is_unavailable_cuda_provider(exc: Exception) -> bool:
    msg = str(exc)
    return "CUDAExecutionProvider" in msg and "is not available" in msg


def clear_fastembed_model_cache(name: str) -> bool:
    cache_root = Path(os.getenv("FASTEMBED_CACHE_PATH", os.path.join(tempfile.gettempdir(), "fastembed_cache")))
    if not cache_root.exists():
        return False

    targets: list[Path] = []
    for model in TextEmbedding.list_supported_models():
        if model.get("model", "").lower() != name.lower():
            continue

        sources = model.get("sources") or {}
        hf_repo = sources.get("hf")
        if hf_repo:
            targets.append(cache_root / f"models--{hf_repo.replace('/', '--')}")

        targets.append(cache_root / name.split("/")[-1])
        targets.append(cache_root / f"fast-{name.split('/')[-1]}")
        targets.append(cache_root / f"{name.split('/')[-1]}.tar.gz")
        targets.append(cache_root / f"fast-{name.split('/')[-1]}.tar.gz")
        break

    cleared = False
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            cleared = True
        elif path.exists():
            path.unlink(missing_ok=True)
            cleared = True
    return cleared


def encode_texts(model: Any, texts: list[str], *, batch_size: int) -> np.ndarray:
    if hasattr(model, "embed"):
        embeddings = list(model.embed(texts, batch_size=batch_size))
        X = np.asarray(embeddings, dtype=np.float32)
    else:
        X = np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    return X


def main():
    ap = argparse.ArgumentParser(description="KMeans clustering for data/raw/export.csv text column using Sentence-BERT")
    ap.add_argument("--input", default=str(DATA_RAW_DIR / "export.csv"), help="Input CSV path")
    ap.add_argument("--text-col", default="text", help="Text column name")
    ap.add_argument("--k", type=int, default=10, help="Number of clusters")
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

    # Kept for report keyword extraction, not for clustering.
    ap.add_argument("--max-features", type=int, default=10000, help="TF-IDF max features for report top terms")
    ap.add_argument("--min-df", type=int, default=5, help="TF-IDF min document frequency for report top terms")
    ap.add_argument("--max-df", type=float, default=1.0, help="TF-IDF max document frequency for report top terms")
    ap.add_argument("--ngram-max", type=int, default=2, help="Max ngram length for report top terms")
    ap.add_argument("--stopwords", default="", help="Path to stopwords file (one word per line)")
    ap.add_argument("--auto-stopwords", type=int, default=0, help="Auto-add N stopwords using lowest-IDF terms for report keywords")
    ap.add_argument("--auto-stopwords-min-df-ratio", type=float, default=0.0, help="Doc frequency ratio threshold for auto stopwords")
    ap.add_argument("--auto-stopwords-max-features", type=int, default=50000, help="Vocab cap for auto-stopword probe vectorizer")
    ap.add_argument("--auto-stopwords-out", default="", help="Write auto stopwords (one per line) to this path")
    ap.add_argument("--no-default-stopwords", action="store_true", help="Disable default stopwords from resources/stopword.txt")

    ap.add_argument("--sample-size", type=int, default=5000, help="Silhouette sample size")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    ap.add_argument("--filter-passes", type=int, default=0, help="How many times to remove clusters smaller than --min-cluster-size before final clustering")
    ap.add_argument("--min-cluster-size", type=int, default=0, help="Treat clusters with size < this as noise during filtering passes (0 disables)")
    ap.add_argument("--stability-seeds", default="7,13,21,37,101", help="Comma-separated seeds for stability ARI")
    ap.add_argument("--top-terms", type=int, default=15, help="Top report terms per cluster")
    ap.add_argument("--examples", type=int, default=5, help="Example texts per cluster")
    ap.add_argument("--output-csv", default=str(DATA_PROCESSED_DIR / "export_clustered.csv"), help="Output CSV with cluster labels")
    ap.add_argument("--report", default=str(REPORTS_DIR / "cluster_report.txt"), help="Output report path")
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
    )
    df2 = prepared.df
    cleaned2 = prepared.cleaned_texts
    dedup_removed = prepared.dedup_removed

    stop_set = load_stopwords(args.stopwords if args.stopwords else None, use_default=(not args.no_default_stopwords)) or set()
    auto_sw = auto_stopwords_from_low_idf(
        cleaned2,
        base_stopwords=stop_set,
        top_n=args.auto_stopwords,
        min_df_ratio=args.auto_stopwords_min_df_ratio,
        max_features=args.auto_stopwords_max_features,
    )
    stop_set |= auto_sw
    if args.auto_stopwords_out:
        ensure_parent_dir(args.auto_stopwords_out)
        with open(args.auto_stopwords_out, "w", encoding="utf-8") as f:
            for w in sorted(auto_sw):
                f.write(w + "\n")
    stop_words = sorted(stop_set) if stop_set else None

    model = load_embedding_model(args.embedding_model, device=args.device)
    embeddings_all = encode_texts(model, cleaned2, batch_size=args.batch_size)

    df2["cluster"] = -1
    df2["filtered_out"] = False
    df2["filtered_pass"] = pd.Series([pd.NA] * len(df2), dtype="Int64")

    active_mask = np.ones(len(df2), dtype=bool)
    filter_history: list[dict[str, object]] = []
    total_passes = max(0, args.filter_passes)
    use_filtering = total_passes > 0 and args.min_cluster_size > 1

    if use_filtering:
        for pass_idx in range(1, total_passes + 1):
            active_idx = np.flatnonzero(active_mask)
            if active_idx.size <= 1:
                filter_history.append({"pass": pass_idx, "active_before": int(active_idx.size), "removed": 0, "small_clusters": [], "stopped": "not_enough_rows"})
                break

            pass_X = embeddings_all[active_idx]
            _, pass_labels, _, actual_k_pass = fit_kmeans_model(pass_X, k=args.k, random_state=args.random_state)
            counts = pd.Series(pass_labels).value_counts()
            small_clusters = sorted([int(c) for c, n in counts.items() if int(n) < args.min_cluster_size])

            if not small_clusters:
                filter_history.append({"pass": pass_idx, "active_before": int(active_idx.size), "removed": 0, "small_clusters": [], "actual_k": actual_k_pass, "stopped": "no_small_clusters"})
                break

            remove_local = np.isin(pass_labels, small_clusters)
            remove_idx = active_idx[remove_local]
            active_mask[remove_idx] = False
            df2.loc[df2.index[remove_idx], "filtered_out"] = True
            df2.loc[df2.index[remove_idx], "filtered_pass"] = pass_idx
            filter_history.append({"pass": pass_idx, "active_before": int(active_idx.size), "removed": int(remove_idx.size), "small_clusters": small_clusters, "actual_k": actual_k_pass})

    final_idx = np.flatnonzero(active_mask)
    if final_idx.size == 0:
        raise SystemExit("No rows left after filtering passes. Lower --min-cluster-size or --filter-passes.")

    X = embeddings_all[final_idx]
    final_texts = [cleaned2[i] for i in final_idx]
    km, labels, kmeans_params, actual_k = fit_kmeans_model(X, k=args.k, random_state=args.random_state)

    stability_seeds = None
    if args.stability_seeds.strip():
        stability_seeds = [int(x.strip()) for x in args.stability_seeds.split(",") if x.strip()]

    ev = evaluate(
        X,
        labels,
        inertia=float(km.inertia_),
        sample_size=args.sample_size,
        random_state=args.random_state,
        stability_seeds=stability_seeds,
        kmeans_params=kmeans_params,
    )

    df2.loc[df2.index[final_idx], "cluster"] = labels
    ensure_parent_dir(args.output_csv)
    df2.to_csv(args.output_csv, index=False)

    keyword_texts = remove_stopwords_from_texts(final_texts, stop_set)
    tops = top_terms_per_cluster(
        keyword_texts,
        labels,
        stop_words=stop_words,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        top_n=args.top_terms,
    )
    ex = exemplar_texts(X, labels, km, final_texts, per_cluster=args.examples)

    counts = pd.Series(labels).value_counts().sort_index()
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
        f.write(f"k={args.k} final_k={actual_k}\n")
        if use_filtering:
            f.write(f"filter_passes={args.filter_passes} min_cluster_size={args.min_cluster_size} filtered_rows={int((~active_mask).sum())}\n")
        else:
            f.write("filter_passes=0 min_cluster_size=0\n")
        if args.auto_stopwords:
            f.write(f"auto_stopwords_added={len(auto_sw)} (requested={args.auto_stopwords}, min_df_ratio={args.auto_stopwords_min_df_ratio})\n")
        f.write(f"inertia={ev.inertia:.4f}\n")
        if ev.silhouette_cosine is not None:
            f.write(f"silhouette_cosine(sample={min(args.sample_size, len(X))})={ev.silhouette_cosine:.4f}\n")
        else:
            f.write("silhouette_cosine=NA\n")
        f.write(f"davies_bouldin={ev.davies_bouldin:.4f} (lower better)\n")
        f.write(f"calinski_harabasz={ev.calinski_harabasz:.4f} (higher better)\n")
        if ev.stability_ari_mean is not None:
            f.write(f"stability_ari_mean={ev.stability_ari_mean:.4f} std={ev.stability_ari_std:.4f}\n")
        else:
            f.write("stability_ari=disabled\n")
        f.write("\n")

        if filter_history:
            f.write("=== Filter Pass Summary ===\n")
            for item in filter_history:
                line = f"pass={item['pass']} active_before={item['active_before']} removed={item['removed']}"
                if "actual_k" in item:
                    line += f" actual_k={item['actual_k']}"
                if item.get("small_clusters"):
                    line += f" small_clusters={item['small_clusters']}"
                if item.get("stopped"):
                    line += f" stopped={item['stopped']}"
                f.write(line + "\n")
            f.write("\n")

        for k in range(actual_k):
            f.write(f"=== Cluster {k} (n={int(counts.get(k, 0))}) ===\n")
            f.write("Top terms: " + ", ".join(tops.get(k, [])) + "\n")
            for i, t in enumerate(ex.get(k, []), start=1):
                t = t.replace("\r", " ").replace("\n", " ").strip()
                if len(t) > 280:
                    t = t[:277] + "..."
                f.write(f"{i}. {t}\n")
            f.write("\n")

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
