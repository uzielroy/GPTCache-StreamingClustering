"""
download_models.py ─ Fetch HF checkpoints into ./models/

Usage examples:
    python download_models.py all              # grab the default shortlist
    python download_models.py tinyllama phi2   # pick specific models
"""

from __future__ import annotations
import sys, time, pathlib
from typing import List
from huggingface_hub import snapshot_download

# ────────────────────────────────────────────────────────────────────────────────
# Open-access models (no authentication needed)
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # MIT
    "falcon":    "tiiuae/falcon-rw-1b",                  # Apache-2.0
    "phi2":      "microsoft/phi-2",                      # MIT
}

TARGET_DIR = pathlib.Path("models").resolve()


def pull(model_id: str, retries: int = 3) -> None:
    """Download/ resume a model snapshot with basic retry logic."""
    print(f"\n📥  Fetching {model_id} …")
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=TARGET_DIR / model_id.split("/")[-1],
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            break  # success
        except Exception as err:
            if attempt == retries:
                raise RuntimeError(
                    f"❌  Failed to download {model_id} after {retries} retries"
                ) from err
            print(f"⚠️  {err} — retry {attempt}/{retries} …")
            time.sleep(4 * attempt)


def main(args: List[str]) -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Decide which models to pull
    if not args or args == ["all"]:
        selection = DEFAULT_MODELS
    else:
        missing = [k for k in args if k not in DEFAULT_MODELS]
        if missing:
            print(f"Unknown model key(s): {', '.join(missing)}")
            sys.exit(1)
        selection = {k: DEFAULT_MODELS[k] for k in args}

    # Download each model
    for key, repo in selection.items():
        pull(repo)

    print("\n✅  All requested models are available under ./models/")


if __name__ == "__main__":
    main(sys.argv[1:])

