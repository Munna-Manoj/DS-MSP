"""Two mistakes in the setup page's source that no server test can see."""
import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web" / "src"


@pytest.mark.req("FR-UI-023")
def test_no_border_colour_also_reads_as_the_left_edge():
    """Tailwind reads ``border-s-<colour>`` as the inline-start border too, so a Studio colour
    whose name after ``s-`` is also a theme colour paints the left edge of a box differently."""
    theme = {n for n in re.findall(r"--color-([a-z0-9-]+):", (WEB / "index.css").read_text(encoding="utf-8"))
             if not n.startswith("s-")}
    used = set()
    for path in WEB.rglob("*.tsx"):
        used |= set(re.findall(r"\bborder-s-([a-z0-9-]+)", path.read_text(encoding="utf-8")))
    assert used, "the pattern no longer finds Studio borders"
    assert not used & theme


@pytest.mark.req("FR-UI-023")
def test_steps_are_numbered_without_a_gap_and_none_is_always_done():
    page = (WEB / "studio" / "NewRunPage.tsx").read_text(encoding="utf-8")
    shown = sorted({int(n) for n in re.findall(r"<Step n=\{(\d+)\}", page)})
    tracked = [int(n) for n in re.findall(r'\{ id: "[a-z]+", n: (\d+),', page)]
    assert shown == tracked == list(range(1, len(tracked) + 1))
    assert not re.search(r'<Step [^>]*state="done"', page)
