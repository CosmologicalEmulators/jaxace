"""Restoration must not silently skip tests or replace modified references."""
import hashlib
import pytest
from . import download_fixtures as loader


def setup_case(tmp_path):
    root = tmp_path / "checkout"
    cache = tmp_path / "artifact"
    name = "tests/data/example.txt"
    source = cache / "jaxace" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"1 2 3\n")
    manifest = tmp_path / "fixtures.sha256"
    manifest.write_text(hashlib.sha256(source.read_bytes()).hexdigest()+"  "+name+"\n")
    return root, cache, source, manifest, root / name


def test_restore_only_data_and_reuse_offline(tmp_path, monkeypatch):
    root, cache, source, manifest, dest = setup_case(tmp_path)
    (cache / "jaxace" / "source.py").write_text("must not be copied")
    monkeypatch.setattr(loader, "artifact", lambda *a, **kw: cache)
    loader.ensure_fixtures(root, manifest)
    assert dest.read_bytes() == source.read_bytes()
    assert not (root / "source.py").exists()
    def offline(*a, **kw):
        raise RuntimeError("network unavailable")
    monkeypatch.setattr(loader, "artifact", offline)
    loader.ensure_fixtures(root, manifest)
    dest.unlink()
    with pytest.raises(RuntimeError, match="network unavailable"):
        loader.ensure_fixtures(root, manifest)


def test_refuse_modified_local_and_cached_data(tmp_path, monkeypatch):
    root, cache, source, manifest, dest = setup_case(tmp_path)
    monkeypatch.setattr(loader, "artifact", lambda *a, **kw: cache)
    dest.parent.mkdir(parents=True)
    dest.write_text("local change")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        loader.ensure_fixtures(root, manifest)
    assert dest.read_text() == "local change"
    dest.unlink()
    source.write_text("corrupt cache")
    with pytest.raises(ValueError, match="Cached fixture checksum mismatch"):
        loader.ensure_fixtures(root, manifest)
    assert not dest.exists()


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "tests/../outside.txt", "source.py"])
def test_reject_unsafe_paths(tmp_path, name):
    manifest = tmp_path / "fixtures.sha256"
    manifest.write_text("0"*64+"  "+name+"\n")
    with pytest.raises(ValueError, match="Unsafe fixture path"):
        loader.ensure_fixtures(tmp_path / "root", manifest)
