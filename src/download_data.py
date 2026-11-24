"""Download and integrity-check the public BNCI Horizon 2020 001-2014 files.

The primary server is the dataset provider. The Hugging Face mirror is used only
as a fallback because the primary server can be temporarily unreachable. Raw
files are never redistributed by this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from scipy.io import loadmat

PRIMARY = "https://bnci-horizon-2020.eu/database/data-sets/001-2014"
MIRROR = (
    "https://hf-mirror.com/datasets/raei/"
    "Brunner-2008-BCICompetition2008GrazDataSetA/resolve/main"
)


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def valid_mat_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        content = loadmat(path, variable_names=["data"], struct_as_record=False)
        # Most files contain three EOG-calibration cells plus six MI runs.
        # A04T has only one EOG-calibration cell (documented technical issue),
        # so its valid top-level data array has seven rather than nine cells.
        return "data" in content and content["data"].size >= 7
    except (OSError, ValueError, TypeError):
        return False


def download_file(url: str, destination: Path, timeout: int = 90) -> None:
    """Stream one file to a temporary path and atomically rename it."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not valid_mat_file(temporary):
                raise ValueError("Downloaded file is not a valid BNCI MAT file")
            temporary.replace(destination)
            return
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_dataset(
    data_dir: Path,
    subjects: list[int],
    source: str = "auto",
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for subject in subjects:
        if not 1 <= subject <= 9:
            raise ValueError("Subjects must be integers from 1 to 9")
        for session in ("T", "E"):
            filename = f"A{subject:02d}{session}.mat"
            destination = data_dir / filename
            if skip_existing and valid_mat_file(destination):
                print(f"[exists] {filename}", flush=True)
            else:
                sources = {
                    "primary": [PRIMARY],
                    "mirror": [MIRROR],
                    "auto": [PRIMARY, MIRROR],
                }[source]
                errors = []
                for base_url in sources:
                    try:
                        print(f"[download] {filename} from {base_url}", flush=True)
                        download_file(f"{base_url}/{filename}", destination)
                        break
                    except RuntimeError as exc:
                        errors.append(str(exc))
                        print(f"[fallback] {exc}", flush=True)
                else:
                    raise RuntimeError("; ".join(errors))
            manifest.append(
                {
                    "filename": filename,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256sum(destination),
                }
            )
    manifest_path = data_dir / "download_manifest.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {entry["filename"]: entry for entry in existing}
    by_name.update({entry["filename"]: entry for entry in manifest})
    manifest_path.write_text(
        json.dumps(list(by_name.values()), indent=2), encoding="utf-8"
    )
    return manifest


def parse_subjects(values: list[str]) -> list[int]:
    """Parse either individual subject numbers or inclusive ranges such as 1-9."""
    subjects: list[int] = []
    for value in values:
        if "-" in value:
            start, end = (int(part) for part in value.split("-", maxsplit=1))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(value))
    return sorted(set(subjects))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--subjects", nargs="+", default=["1-9"])
    parser.add_argument("--source", choices=["auto", "primary", "mirror"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = download_dataset(
        args.data_dir,
        parse_subjects(args.subjects),
        source=args.source,
        skip_existing=not args.overwrite,
    )
    print(f"Validated {len(manifest)} files. Manifest: {args.data_dir / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
