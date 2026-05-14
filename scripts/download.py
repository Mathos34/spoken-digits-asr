"""Clone the Free Spoken Digit Dataset (FSDD) recordings into ./data/recordings/."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _rm_onerror(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDINGS = DATA / "recordings"
URL = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"


def main() -> None:
    if RECORDINGS.exists() and any(RECORDINGS.glob("*.wav")):
        n = sum(1 for _ in RECORDINGS.glob("*.wav"))
        print(f"FSDD already present: {n} wav files in {RECORDINGS}")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = DATA / "_fsdd_repo"
    if tmp.exists():
        shutil.rmtree(tmp, onerror=_rm_onerror)
    print(f"Cloning FSDD into {tmp} (this is ~50 MB)...")
    subprocess.check_call(["git", "clone", "--depth", "1", URL, str(tmp)])
    src_rec = tmp / "recordings"
    if not src_rec.exists():
        print(f"Error: {src_rec} not found in cloned repo", file=sys.stderr)
        sys.exit(1)
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    for wav in src_rec.glob("*.wav"):
        shutil.copy(wav, RECORDINGS / wav.name)
    shutil.rmtree(tmp, onerror=_rm_onerror)
    n = sum(1 for _ in RECORDINGS.glob("*.wav"))
    print(f"Done. {n} wav files in {RECORDINGS}")


if __name__ == "__main__":
    main()
