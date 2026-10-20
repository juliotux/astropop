"""Keep Sphinx implementation modules outside ordinary doctest collection."""

from pathlib import Path

import pytest


# pytest-doctestplus imports Python modules while collecting documentation.
# These helpers require the docs extra; the Sphinx build validates them.
collect_ignore = ['_ext']


@pytest.fixture(autouse=True)
def tutorial_helpers(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Provide tutorial helpers and restore optional-array settings after each page."""
    if request.node.path.parent.name != 'tutorials':
        return
    from astropop.config import AstropopConfig

    monkeypatch.syspath_prepend(str(Path(__file__).parent / '_ext'))
    monkeypatch.setattr(AstropopConfig, 'FRAMEDATA_DISABLE_UNCERTAINTY', False)
    monkeypatch.setattr(AstropopConfig, 'FRAMEDATA_DISABLE_FLAGS', False)
