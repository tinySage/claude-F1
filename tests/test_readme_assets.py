"""README SVGs are valid, self-contained and come in both themes."""

import xml.etree.ElementTree as ET

import pytest

from f1strat import readme_assets as ra


@pytest.mark.parametrize("theme", list(ra.THEMES))
def test_verdict_cards_are_valid_svg_without_web_fonts(theme: str) -> None:
    svg = ra.verdict_cards(theme)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "@font-face" not in svg and "http" not in svg.replace("http://www.w3.org", "")
    text = " ".join(t.text or "" for t in root.iter() if t.tag.endswith("text"))
    for label in ("Within range", "Outside range", "Inconclusive", "Not assessable at the time"):
        assert label in text
    assert text.count("HINDSIGHT") == 4


def test_themes_share_tokens() -> None:
    assert set(ra.THEMES["dark"]) == set(ra.THEMES["light"])
