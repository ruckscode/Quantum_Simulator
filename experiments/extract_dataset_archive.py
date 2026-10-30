"""Safely extract ZIP archives, mapping Windows-invalid names and recording mappings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def portable_part(part: str) -> str:
    bad = '<>:"|?*'
    cooked = ''.join(f'%{ord(ch):02X}' if ch in bad else ch for ch in part)
    while cooked.endswith((' ', '.')):
        cooked = cooked[:-1] + f'%{ord(cooked[-1]):02X}'
    if cooked.split('.')[0].upper() in {
        'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
        *(f'LPT{i}' for i in range(1, 10)),
    }:
        cooked = '%' + cooked
    return cooked or '%00'


def extract(archive: Path, destination: Path) -> dict:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root = str(destination)
    if os.name == 'nt' and not root.startswith('\\\\?\\'):
        root = '\\\\?\\' + root
    mappings = []
    count = 0
    bytes_written = 0
    def destination_for(name: str) -> tuple[list[str], str]:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe archive member: {name!r}')
        parts = [portable_part(part) for part in relative.parts]
        return parts, os.path.join(root, *parts)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            regular = [info for info in bundle.infolist() if not info.is_dir()]
            cooked_names = {}
            folded = {}
            for info in regular:
                parts = [portable_part(part) for part in PurePosixPath(info.filename).parts]
                cooked_names[info.filename] = parts
                folded.setdefault('/'.join(parts).casefold(), []).append(info.filename)
            # Windows cannot represent paths that differ only by case. Keep the
            # lexicographically first archive path unchanged and encode the
            # colliding filename with a stable suffix; record every rename.
            for names in folded.values():
                if len(names) < 2:
                    continue
                for name in sorted(names)[1:]:
                    parts = cooked_names[name]
                    stem, ext = os.path.splitext(parts[-1])
                    suffix = hashlib.sha256(name.encode('utf-8')).hexdigest()[:10]
                    parts[-1] = f'{stem}~case-{suffix}{ext}'
            for info in bundle.infolist():
                if info.is_dir():
                    parts, target = destination_for(info.filename)
                else:
                    parts = cooked_names[info.filename]
                    target = os.path.join(root, *parts)
                if info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with bundle.open(info) as source, open(target, 'wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                count += 1
                bytes_written += info.file_size
                if parts != list(PurePosixPath(info.filename).parts):
                    mappings.append({'archive_path': info.filename, 'extracted_path': '/'.join(parts)})
    else:
        with tarfile.open(archive, mode='r|') as bundle:
            for info in bundle:
                parts, target = destination_for(info.name)
                if info.isdir():
                    os.makedirs(target, exist_ok=True)
                    continue
                if not info.isfile():
                    raise ValueError(f'Unsupported non-regular tar member: {info.name!r}')
                os.makedirs(os.path.dirname(target), exist_ok=True)
                source = bundle.extractfile(info)
                if source is None:
                    raise OSError(f'Could not read tar member: {info.name!r}')
                with source, open(target, 'wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                count += 1
                bytes_written += info.size
                if parts != list(PurePosixPath(info.name).parts):
                    mappings.append({'archive_path': info.name, 'extracted_path': '/'.join(parts)})
    manifest = {
        'archive': str(archive),
        'destination': str(destination),
        'files_extracted': count,
        'uncompressed_bytes': bytes_written,
        'path_mappings': mappings,
    }
    (destination / '_extraction_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.destination), indent=2))
