import pytest


@pytest.fixture(autouse=True)
def _isolated_yc_cache(tmp_path, monkeypatch):
    """Keep tests hermetic: a real YC dataset cached by a live run (data/cache) must not feed companies into mocks."""
    monkeypatch.setattr("src.sources.companies.YC_CACHE_PATH", str(tmp_path / "yc_companies.json"))
