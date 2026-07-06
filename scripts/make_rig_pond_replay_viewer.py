#!/usr/bin/env python3
"""
Build docs/assets/rig_pond_replay/index.html: a static replay of the live rig-calibration pond
viewer (ds_msp.rig.web3d.WebLive3DAnimator's _VIEWER_HTML), for embedding in the published docs
/ GitHub Pages (which cannot run the live Python server behind the real thing).

Reuses the exact, real, tested _VIEWER_HTML byte-for-byte (imported from the module, not
retyped) -- every rendering/animation function (pond, fish, director acts, digital-twin finale)
is identical to the live view. The only change: the live view's poll() polls a running Python
server for state.json every 150ms; the replay instead fetches the pre-recorded, gzip-compressed
frames.json.gz (see scripts/make_rig_pond_replay.py), decompresses it client-side with the
browser's native DecompressionStream('gzip') (no external JS library), and steps through the
frames on a timer -- everything downstream (onState, updateOrbs, the finale reveal) is the same
code driven by the same shape of state object, so nothing else needs to change.

    python scripts/make_rig_pond_replay_viewer.py
"""
from __future__ import annotations

import os

from ds_msp.rig.web3d import _VIEWER_HTML

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(HERE, "docs", "assets", "rig_pond_replay", "index.html")

_LIVE_POLL_BLOCK = """async function poll() {
  try {
    const res = await fetch('state.json', { cache: 'no-store' });
    if (res.ok) onState(await res.json());
  } catch (e) { /* server may be gone after the run exits -- last state already shown */ }
  if (!finished) setTimeout(poll, 150);
}
initFish();
poll();"""

_REPLAY_BLOCK = """// --- STATIC REPLAY --------------------------------------------------------------------
// This page has no backend: it replays one real, recorded calibration run (see
// docs/RIG_CALIBRATION_GUIDE.md §7) by stepping through frames.json.gz instead of
// polling a live Python server. Everything downstream (onState, updateOrbs, the finale
// reveal) is the exact same code the live view uses.
const REPLAY_FRAME_MS = 90;
let replayFrames = [], replayIdx = 0;

async function loadReplayFrames() {
  const res = await fetch('frames.json.gz');
  const text = await new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).text();
  return JSON.parse(text);
}

function stepReplay() {
  if (replayIdx < replayFrames.length) {
    onState(replayFrames[replayIdx]);
    replayIdx += 1;
  } else {
    finished = true;
  }
  if (!finished) setTimeout(stepReplay, REPLAY_FRAME_MS);
}

loadReplayFrames().then(frames => {
  replayFrames = frames;
  initFish();
  stepReplay();
}).catch(e => {
  document.getElementById('status').textContent = 'failed to load replay: ' + e;
});"""

_REPLAY_BADGE_CSS = """
  #replayBadge { position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
                 background: rgba(4,20,26,.82); border: 1px solid #1e4a56; border-radius: 999px;
                 padding: 6px 16px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
                 color: #7dd3fc; font-weight: 700; z-index: 5; pointer-events: none; }"""

_REPLAY_BADGE_HTML = """
<div id="replayBadge">Replay of a real recorded run</div>"""


def main() -> None:
    html = _VIEWER_HTML
    assert _LIVE_POLL_BLOCK in html, "live poll() block not found -- _VIEWER_HTML changed shape"
    html = html.replace(_LIVE_POLL_BLOCK, _REPLAY_BLOCK)
    html = html.replace("<title>DS-MSP -- live rig calibration</title>",
                         "<title>DS-MSP -- rig calibration replay</title>")
    html = html.replace("</style>\n</head>", _REPLAY_BADGE_CSS + "\n</style>\n</head>")
    html = html.replace('<canvas id="cv"></canvas>',
                         '<canvas id="cv"></canvas>' + _REPLAY_BADGE_HTML)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes)")


if __name__ == "__main__":
    main()
