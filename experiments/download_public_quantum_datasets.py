"""Resumable acquisition helper for public quantum software research artifacts.

Examples:
  python experiments/download_public_quantum_datasets.py qolumbina_empirical
  python experiments/download_public_quantum_datasets.py lintq_programs
  python experiments/download_public_quantum_datasets.py bugs4q_framework

All destinations are under data/external_quantum_datasets. The script never
uses credentials and verifies published archive checksums where available.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/external_quantum_datasets'

REPOSITORIES = {
    'bugs4q_framework': ('https://github.com/Z-928/Bugs4Q-Framework.git', DATA / 'bugs4q_full/official_framework', False),
    'bugs4q_database': ('https://github.com/Z-928/Bugs4Q.git', DATA / 'bugs4q_full/official_database', False),
    'bugs4q_na': ('https://github.com/Z-928/Bugs4Q-NA.git', DATA / 'bugs4q_na', False),
    'bugs4q_robust': ('https://github.com/kusumotolab/quantum-defect-reproducibility.git', DATA / 'bugs4q_robust', False),
    'qsimbench_library': ('https://github.com/GBisi/qsimbench.git', DATA / 'qsimbench/library', False),
    'qsimbench_dataset': ('https://github.com/GBisi/qsimbench-dataset.git', DATA / 'qsimbench/dataset', False),
    'quantum_platform_bugs_223': ('https://github.com/MattePalte/Bugs-Quantum-Computing-Platforms.git', DATA / 'quantum_platform_bugs_223', True),
    'qml_platform_bugs': ('https://github.com/Z-928/QML-platform-bugs.git', DATA / 'qml_platform_bugs', True),
    'lintq_code': ('https://github.com/sola-st/LintQ.git', DATA / 'lintq', False),
    'bug_pattern': ('https://github.com/Z-928/Bug-Pattern.git', DATA / 'bug_pattern', False),
    'qchecker': ('https://github.com/Z-928/QChecker.git', DATA / 'qchecker', False),
}

ARCHIVES = {
    'qolumbina_empirical': {
        'url': 'https://zenodo.org/records/20987445/files/qolumbina-empirical-study-main.zip?download=1',
        'path': DATA / 'qolumbina/qolumbina-empirical-study-main.zip',
        'md5': '3995d9dd9a2031e66fa561722e9e58fe',
    },
    'lintq_programs': {
        'url': 'https://ndownloader.figshare.com/files/45988725',
        'path': DATA / 'lintq/raw/artifact_lintq_dataset.tar',
        'md5': 'f49394cca97448a40ea0fdca674f7b9a',
    },
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> Path:
    spec = ARCHIVES[name]
    path: Path = spec['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path, 'md5') == spec['md5']:
        print(f'Verified existing archive: {path}')
        return path
    part = path.with_suffix(path.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    headers = {'User-Agent': 'QuantumResearchDatasetAcquirer/1.0'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(spec['url'], headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Download failed for {name}; resumable partial at {part}: {exc}') from exc
    if offset and response.status != 206:
        offset = 0
    mode = 'ab' if offset else 'wb'
    last_report = time.monotonic()
    received = offset
    with response, part.open(mode) as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
            received += len(block)
            if time.monotonic() - last_report > 10:
                print(f'{name}: {received:,} bytes downloaded', flush=True)
                last_report = time.monotonic()
    actual = digest(part, 'md5')
    if actual != spec['md5']:
        raise ValueError(f'{name} checksum mismatch: expected {spec["md5"]}, got {actual}; partial retained at {part}')
    part.replace(path)
    print(f'{name}: verified {path.stat().st_size:,} bytes; MD5 {actual}')
    return path


def clone(name: str) -> Path:
    url, destination, needs_archive = REPOSITORIES[name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / '.git').exists():
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()
        print(f'Existing Git repository: {destination}; HEAD {commit}')
        if needs_archive:
            archive_objects(destination)
        return destination
    if needs_archive:
        # These source trees contain names that Win32 cannot materialize. Keep
        # all blobs in Git and produce a ZIP with the original archive paths.
        subprocess.run(['git', 'clone', '--depth', '1', '--no-checkout', url, str(destination)], check=True)
        archive_objects(destination)
        return destination
    subprocess.run(['git', 'clone', '--depth', '1', url, str(destination)], check=True)
    print(f'{name}: cloned HEAD {subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()}')
    return destination


def archive_objects(repository: Path) -> Path:
    """Write a ZIP directly from Git blobs, preserving names Windows cannot check out."""
    archive = repository.parent / f'{repository.name}_source.zip'
    if archive.exists() and archive.stat().st_size:
        print(f'Existing source archive: {archive}')
        return archive
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', '--full-tree', 'HEAD'], cwd=repository)
    entries = []
    for row in tree.split(b'\0'):
        if not row:
            continue
        meta, raw_path = row.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        if kind == 'blob':
            entries.append((oid, raw_path.decode('utf-8', 'surrogateescape'), int(mode, 8)))
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=repository,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def feed() -> None:
        assert process.stdin is not None
        try:
            for oid, _, _ in entries:
                process.stdin.write((oid + '\n').encode())
        finally:
            process.stdin.close()
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    import zipfile
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as output:
        assert process.stdout is not None
        for _, name, mode in entries:
            header = process.stdout.readline().rstrip(b'\n').split()
            if len(header) != 3 or header[1] != b'blob':
                raise RuntimeError(f'Unexpected Git blob header: {header!r}')
            remaining = int(header[2])
            info = zipfile.ZipInfo(name)
            info.external_attr = (mode & 0xffff) << 16
            with output.open(info, 'w') as target:
                while remaining:
                    block = process.stdout.read(min(1024 * 1024, remaining))
                    if not block:
                        raise EOFError(f'Git blob ended early: {name}')
                    target.write(block)
                    remaining -= len(block)
            if process.stdout.read(1) != b'\n':
                raise RuntimeError(f'Missing Git blob delimiter: {name}')
    feeder.join()
    if process.wait():
        error = process.stderr.read().decode(errors='replace') if process.stderr else ''
        raise RuntimeError(f'Could not create source archive: {error}')
    print(f'Wrote original-path source archive: {archive} ({archive.stat().st_size:,} bytes)')
    return archive


def main() -> None:
    choices = sorted((*REPOSITORIES, *ARCHIVES))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=choices)
    args = parser.parse_args()
    if args.source in ARCHIVES:
        download(args.source)
    else:
        clone(args.source)


if __name__ == '__main__':
    main()
