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
    return command
