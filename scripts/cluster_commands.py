import shlex
from collections.abc import Mapping


def _as_bool(value: object) -> bool:
    return bool(value)


def _as_str(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_int(value: object, *, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: object, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def build_base_args(config: Mapping[str, object]) -> list[str]:
    command = [
        "--input",
        _as_str(config.get("input")),
        "--text-col",
        _as_str(config.get("text_col"), default="text"),
        "--min-len",
        str(_as_int(config.get("min_len"), default=10)),
        "--min-words",
        str(_as_int(config.get("min_words"), default=5)),
        "--batch-size",
        str(_as_int(config.get("batch_size"), default=64)),
        "--sample-size",
        str(_as_int(config.get("sample_size"), default=5000)),
    ]
    if _as_int(config.get("limit"), default=0) > 0:
        command.extend(["--limit", str(_as_int(config.get("limit"), default=0))])
    if _as_bool(config.get("limit_random")):
        command.append("--limit-random")
    if _as_bool(config.get("drop_duplicate_texts")):
        command.append("--drop-duplicate-texts")
    if _as_str(config.get("embedding_model")):
        command.extend(["--embedding-model", _as_str(config.get("embedding_model"))])
    if _as_str(config.get("device")):
        command.extend(["--device", _as_str(config.get("device"))])
    if _as_bool(config.get("keep_hashtags")):
        command.append("--keep-hashtags")
    if _as_bool(config.get("remove_emojis")):
        command.append("--remove-emojis")
    if _as_str(config.get("stopwords")):
        command.extend(["--stopwords", _as_str(config.get("stopwords"))])
    if _as_bool(config.get("no_default_stopwords")):
        command.append("--no-default-stopwords")
    return command


def build_kmeans_command(python_bin: str, config: Mapping[str, object]) -> list[str]:
    command = [python_bin, "scripts/kmeans_text_cluster.py", *build_base_args(config)]
    command.extend(
        [
            "--k",
            str(_as_int(config.get("k"), default=10)),
            "--output-csv",
            _as_str(config.get("kmeans_output")),
            "--report",
            _as_str(config.get("kmeans_report")),
        ]
    )
    if _as_str(config.get("kmeans_extra")).strip():
        command.extend(shlex.split(_as_str(config.get("kmeans_extra"))))
    return command


def build_dbscan_command(python_bin: str, config: Mapping[str, object]) -> list[str]:
    command = [python_bin, "scripts/dbscan_text_cluster.py", *build_base_args(config)]
    command.extend(
        [
            "--eps",
            str(_as_float(config.get("eps"), default=0.85)),
            "--min-samples",
            str(_as_int(config.get("min_samples"), default=10)),
            "--output-csv",
            _as_str(config.get("dbscan_output")),
            "--report",
            _as_str(config.get("dbscan_report")),
        ]
    )
    if _as_str(config.get("dbscan_extra")).strip():
        command.extend(shlex.split(_as_str(config.get("dbscan_extra"))))
    if _as_bool(config.get("dbscan_use_umap")):
        command.extend(
            [
                "--use-umap",
                "--umap-components",
                str(_as_int(config.get("dbscan_umap_components"), default=10)),
                "--umap-neighbors",
                str(_as_int(config.get("dbscan_umap_neighbors"), default=15)),
                "--umap-min-dist",
                str(_as_float(config.get("dbscan_umap_min_dist"), default=0.0)),
            ]
        )
    return command


def build_hdbscan_command(python_bin: str, config: Mapping[str, object]) -> list[str]:
    command = [python_bin, "scripts/hdbscan_text_cluster.py", *build_base_args(config)]
    command.extend(
        [
            "--min-cluster-size",
            str(_as_int(config.get("hdbscan_min_cluster_size"), default=15)),
            "--min-samples",
            str(_as_int(config.get("hdbscan_min_samples"), default=5)),
            "--cluster-selection-epsilon",
            str(_as_float(config.get("hdbscan_cluster_selection_epsilon"), default=0.0)),
            "--output-csv",
            _as_str(config.get("hdbscan_output")),
            "--report",
            _as_str(config.get("hdbscan_report")),
        ]
    )
    if _as_str(config.get("hdbscan_extra")).strip():
        command.extend(shlex.split(_as_str(config.get("hdbscan_extra"))))
    if _as_bool(config.get("hdbscan_use_umap")):
        command.extend(
            [
                "--use-umap",
                "--umap-components",
                str(_as_int(config.get("hdbscan_umap_components"), default=10)),
                "--umap-neighbors",
                str(_as_int(config.get("hdbscan_umap_neighbors"), default=15)),
                "--umap-min-dist",
                str(_as_float(config.get("hdbscan_umap_min_dist"), default=0.0)),
            ]
        )
    return command
