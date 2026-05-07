import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent

CACHE_DIR = PROJECT_ROOT / ".cache"
TMP_DIR = CACHE_DIR / "tmp"
PIP_CACHE_DIR = CACHE_DIR / "pip"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
FASTEMBED_CACHE_DIR = CACHE_DIR / "fastembed"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESOURCES_DIR = PROJECT_ROOT / "resources"


def ensure_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def configure_local_runtime_storage() -> None:
    for path in (CACHE_DIR, TMP_DIR, PIP_CACHE_DIR, HF_CACHE_DIR, FASTEMBED_CACHE_DIR):
        ensure_dir(path)

    os.environ.setdefault("TMPDIR", str(TMP_DIR))
    os.environ.setdefault("TMP", str(TMP_DIR))
    os.environ.setdefault("TEMP", str(TMP_DIR))
    os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(FASTEMBED_CACHE_DIR))

    # Force Python's tempfile module to use the project-local temp directory.
    tempfile.tempdir = str(TMP_DIR)


configure_local_runtime_storage()
