# View rig results in DS-MSP Studio

DS-MSP Studio shows rig calibration results in your browser. It reads the folders that
`ds-msp-calibrate-rig` writes and never changes them.

## Install and start

```bash
pip install "ds-msp[ui]"
ds-msp studio --root path/to/results
```

Studio searches `--root` (and up to six folder levels below it) for result folders, that is,
folders holding `calibration_report.json`. It prints its address, `http://127.0.0.1:8765/`, and
opens it. Pass `--root` more than once to search several folders, and `--no-browser` to skip
opening a browser.

## On a lab server

Studio listens on `127.0.0.1` only, so forward the port over SSH:

```bash
ssh -L 8765:127.0.0.1:8765 lab-server
ds-msp studio --root /data/calibration --no-browser     # on the server
```

Then open `http://127.0.0.1:8765/` on your own computer. If you bind to another address with
`--host`, Studio requires a token and prints a URL that carries it.

## What the result page shows

- **Banner.** How many cameras are ready, the typical reprojection error, and whether any camera
  needs attention.
- **Cameras.** One tile per camera:
  - *Ready*: no recorded check flagged it.
  - *Needs photos*: the lens check could not bound its lens, a lens parameter sits at its
    limit, or the fit explains too few of its corners.
  - *Check*: the observability audit found it weakly linked to the rest of the rig, or the
    opt-in split check disagreed.
- **Rig.** The cameras in 3D, in the reference camera's frame. Drag to turn, scroll to zoom.
- **Camera details.** Photos used, typical and largest error, lens uncertainty, the range of
  board tilt it saw, and its lens parameters with one-sigma values.
- **How well cameras agree.** The one-sigma rotation and position uncertainty of every pair,
  weakest first.
- **Technical details.** Solver stages, the checks that ran, the resolved configuration, and
  links to every result file.

/// warning | Uncertainties are lower bounds
Every ± value comes from photo noise alone. On real captures the true error has been 2 to 3
times larger, so Studio writes them as "at least". To validate a calibration, take a second,
separate round of board photos and check the calibration against it. Studio labels every result
"not validated with a second capture" until that step exists.
///

## Older results

Results written before the per-camera lens checks have no lens or split-check fields. Studio
shows those values as "not recorded" instead of marking the camera as failed.

## Build the interface from a source checkout

Released wheels include the built interface. In a git checkout, build it once:

```bash
cd web
npm ci
npm run build:studio      # writes ds_msp/studio/static
```

Without it, `ds-msp studio` serves a page that explains this, and the JSON API at
`/api/runs` still works.
