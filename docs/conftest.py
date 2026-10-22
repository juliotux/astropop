"""Keep Sphinx implementation modules outside ordinary doctest collection."""

from pathlib import Path

import pytest


# pytest-doctestplus imports Python modules while collecting documentation.
# These helpers require the docs extra; the Sphinx build validates them.
collect_ignore = ['_ext', '_build']


@pytest.fixture(autouse=True)
def tutorial_helpers(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
                     tmp_path: Path) -> None:
    """Provide tutorial helpers and restore optional-array settings after each page."""
    if not request.node.path.is_relative_to(Path(__file__).parent / 'tutorials'):
        return
    monkeypatch.chdir(tmp_path)
    from astropop.config import AstropopConfig

    monkeypatch.syspath_prepend(str(Path(__file__).parent / '_ext'))
    monkeypatch.setattr(AstropopConfig, 'FRAMEDATA_DISABLE_UNCERTAINTY', False)
    monkeypatch.setattr(AstropopConfig, 'FRAMEDATA_DISABLE_FLAGS', False)
