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


@pytest.mark.req("FR-UI-024")
def test_a_single_board_can_be_given_its_printed_number():
    """MC-Calib's single-board sequences use board 12 or 13 of a printed set; without a place to
    say so, no board is found."""
    step = (WEB / "studio" / "setup" / "BoardStep.tsx").read_text(encoding="utf-8")
    at = step.index('<KeyTag name="boards_index" />')
    start = step.rindex('id="board-printed"', 0, at)
    assert "nb > 1 &&" not in step[start:at]


@pytest.mark.req("FR-UI-024")
def test_a_view_that_fails_to_draw_does_not_blank_the_page():
    stage = (WEB / "studio" / "stage" / "Stage.tsx").read_text(encoding="utf-8")
    assert stage.index("<Guard") < stage.index("<RigStage") < stage.index("</Guard>")
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    assert app.index("<Guard") < app.index("<NewRunPage") < app.index("</Guard>")
