from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

import grpc_tools


WAYMO_REVISION = "99a4cb3ff07e2fe06c2ce73da001f850f628e45a"
BASE_URL = (
    "https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/"
    f"{WAYMO_REVISION}/src/"
)
ENTRY_PROTO = "waymo_open_dataset/protos/end_to_end_driving_data.proto"
IMPORT_PATTERN = re.compile(r'^\s*import\s+"([^"]+)"\s*;', re.MULTILINE)


def _safe_relative_path(import_path: str) -> PurePosixPath:
    relative = PurePosixPath(import_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe protobuf import path: {import_path}")
    return relative


def _download_proto(import_path: str) -> bytes:
    relative = _safe_relative_path(import_path)
    url = BASE_URL + relative.as_posix()
    request = urllib.request.Request(url, headers={"User-Agent": "honours-project"})
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = response.geturl()
        if not final_url.startswith(BASE_URL):
            raise RuntimeError(f"refused protobuf redirect outside pinned source: {final_url}")
        return response.read()


def _fetch_tree(import_path: str, source_root: Path, fetched: set[str]) -> None:
    if import_path in fetched or import_path.startswith("google/protobuf/"):
        return

    payload = _download_proto(import_path)
    relative = _safe_relative_path(import_path)
    destination = source_root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    fetched.add(import_path)

    text = payload.decode("utf-8")
    for dependency in IMPORT_PATTERN.findall(text):
        _fetch_tree(dependency, source_root, fetched)


def bootstrap(output_root: Path) -> list[Path]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="waymo-protos-") as temporary:
        source_root = Path(temporary) / "src"
        source_root.mkdir()
        fetched: set[str] = set()
        _fetch_tree(ENTRY_PROTO, source_root, fetched)

        grpc_include = Path(grpc_tools.__file__).resolve().parent / "_proto"
        command = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{source_root}",
            f"-I{grpc_include}",
            f"--python_out={output_root}",
            *sorted(str(source_root / path) for path in fetched),
        ]
        subprocess.run(command, check=True)

    generated_root = output_root / "waymo_open_dataset"
    for directory in [generated_root, *(path for path in generated_root.rglob("*") if path.is_dir())]:
        (directory / "__init__.py").touch(exist_ok=True)
    return sorted(generated_root.rglob("*_pb2.py"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Python protobuf modules for the Waymo E2E format."
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    generated = bootstrap(args.output_root)
    if not generated:
        raise RuntimeError("Waymo protobuf generation produced no Python modules")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
