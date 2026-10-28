"""Restore only checksum-listed test data, never archived source code."""
import hashlib
from pathlib import Path
import shutil
import tempfile

from fetch_artifacts import artifact

HERE = Path(__file__).resolve().parent


def ensure_fixtures(root=HERE.parent, manifest=HERE / "fixtures.sha256"):
    root = Path(root)
    missing = []
    for line in Path(manifest).read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not name.startswith("tests/"):
            raise ValueError(f"Unsafe fixture path: {name}")
        destination = root / relative
        if destination.is_symlink():
            raise ValueError(f"Fixture must not be a symlink: {name}")
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Fixture checksum mismatch: {name}; refusing to overwrite local data")
        else:
            missing.append((expected, relative))
    if not missing:
        return
    # The existing artifact client checks the archive SHA256 and caches by tree.
    source_root = artifact("reference_fixtures", toml_path=HERE / "Artifacts.toml") / "jaxace"
    for expected, relative in missing:
        source = source_root / relative
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached fixture checksum mismatch: {relative}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_fixtures()
    print("Zenodo reference fixtures verified and available.")
