import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional

from bmds.config import DATA_RAW_DIR


BALABIT_REPO_URL = "https://github.com/balabit/Mouse-Dynamics-Challenge"
BALABIT_ZIP_URL = "https://github.com/balabit/Mouse-Dynamics-Challenge/archive/refs/heads/master.zip"

BALABIT_DIR = DATA_RAW_DIR / "balabit"


def download_balabit(output_dir: Optional[Path] = None, force: bool = False) -> Path:
    import requests

    if output_dir is None:
        output_dir = BALABIT_DIR

    training_dir = output_dir / "training_files"
    if training_dir.exists() and not force:
        n_users = len(list(training_dir.iterdir())) if training_dir.is_dir() else 0
        if n_users > 0:
            print(f"Balabit dataset already exists at {output_dir} ({n_users} users). "
                  "Use force=True to re-download.")
            return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "balabit.zip"

    print(f"Downloading Balabit dataset from {BALABIT_ZIP_URL} ...")
    response = requests.get(BALABIT_ZIP_URL, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024}KB)", end="", flush=True)
    print()

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


    extracted_subdir = output_dir / "Mouse-Dynamics-Challenge-master"
    if extracted_subdir.exists():
        for item in extracted_subdir.iterdir():
            dest = output_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        extracted_subdir.rmdir()


    zip_path.unlink()

    n_users = len(list(training_dir.iterdir())) if training_dir.exists() else 0
    print(f"Balabit dataset ready at {output_dir} ({n_users} user directories)")
    return output_dir


def get_session_files(dataset_dir: Optional[Path] = None):
    if dataset_dir is None:
        dataset_dir = BALABIT_DIR

    training_dir = dataset_dir / "training_files"
    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training files not found at {training_dir}. "
            "Run download_balabit() first."
        )

    sessions = []
    for user_dir in sorted(training_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        for session_file in sorted(user_dir.iterdir()):
            if session_file.is_file():
                sessions.append((user_id, session_file))

    return sessions


if __name__ == "__main__":
    download_balabit()
