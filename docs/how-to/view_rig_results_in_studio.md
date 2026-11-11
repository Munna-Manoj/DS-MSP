# View rig results in DS-MSP Studio

DS-MSP Studio shows a rig calibration result in your browser: a verdict, what to do next, and
the evidence for each camera. It reads the folders that `ds-msp-calibrate-rig` writes and never
changes them.

To set up and start a calibration from the browser instead, see
[Calibrate a rig in DS-MSP Studio](calibrate_a_rig_in_studio.md).

## Install

```bash
pip install "ds-msp[ui]"
```

## Open a result right after calibrating

Add `--studio` to the calibration command:

```bash
ds-msp-calibrate-rig --config calib_param.yml --studio
```

When the calibration is done, the terminal prints the report, then Studio's address, and opens
this result's page in your browser:

```text
report saved: out/calibration_report.txt  +  out/calibration_report.json
run manifest: out/calibration_manifest.json
view in DS-MSP Studio:  ds-msp studio --open "/path/to/rig/out"

DS-MSP Studio: http://127.0.0.1:8765/#/run/0%3Aout
Press Ctrl+C to stop.
```

Studio keeps serving until you press Ctrl+C. The command still exits with the calibration's own
code. If port 8765 is taken, Studio uses the next free one and prints it.

Without `--studio`, every saved calibration ends with the `view in DS-MSP Studio:` line. Run that
command at any time to open the result. Without the `ui` extra, the line also says how to install
it.

## Browse several results

```bash
ds-msp studio --root path/to/results
```

Studio searches `--root`, and up to six folder levels below it, for result folders. A result
folder is one that holds `calibration_report.json`. It prints its address,
`http://127.0.0.1:8765/`, and opens it. Pass `--root` more than once to search several folders,
and `--no-browser` to skip opening a browser.

## On another computer

Studio listens on `127.0.0.1` only. If the results are on another computer that you reach
over SSH, forward the port:

```bash
ssh -L 8765:127.0.0.1:8765 user@host
ds-msp studio --root path/to/results --no-browser     # on that computer
```

Then open `http://127.0.0.1:8765/` on your own computer. If you bind to another address with
`--host`, Studio requires a token and prints a URL that carries it.

## Read a result

A result page answers three questions in order: can I use it, what should I do, and why.

### 1. The verdict

The banner gives one verdict for the rig, and each camera carries its own level, shown as the
dot on its button:

| Level | Meaning |
|---|---|
| **Ready** | Every check the calibration recorded passed. |
| **Check** | Usable; one item is worth a look, such as more than 3% of a camera's corners disagreeing with the fit. |
| **Improve** | Usable, but more photos would help: a lens is uncertain, a lens parameter sits at its limit, a camera is weakly tied to the rig, or the split check disagreed. |
| **Not ready** | A hard check failed: a lens or a camera pair is not determined by the photos, or the fit explains too few of a camera's corners. |

Only checks recorded in `calibration_acceptance.json` set a level. The 3% outlier rule comes
from mrcal and is not yet validated for DS-MSP, so it can only raise *Check*.

### 2. Next steps

At most three actions, worst first. Each names its cameras and links to the view that supports
it, for example:

```text
Improve  Add photos of camera 3 with the board in the areas marked on its uncertainty map,
         close and tilted about 45°.            cam03 · see uncertainty map
```

Select **Show all** for the rest.

### 3. Explore the result on your own capture

The dark panel shows the calibration on the capture itself. Choose a camera with the buttons at
its top. The strip at its bottom is the capture: one bar per photo, as tall as that photo's
error. Click a bar, drag along the strip, or press play; every view follows.

- **On your photos.** The photo, with each corner where it was found (white dot) and a line to
  where the calibration expects it (ring). The line is stretched, by 136 times on a rig with a
  0.05 px typical error, so a sub-pixel error can be seen. Short lines in all directions are
  good. **Shade where this lens was never checked** stripes the parts of the image no board ever
  covered.
- **The rig in 3D.** The cameras and the boards where they were in the selected photo, each
  outlined and named in its own colour, the same colour as on the photos.
  Click a second camera to measure to it: distance, position and yaw, pitch and roll, with how
  well each is known. **Show the uncertainty** makes each camera sway by its own uncertainty,
  enlarged until it is visible.
- **What it means.** For cameras that do not share a view, such as two facing each other, how
  exactly a point known to one is placed for the other. Otherwise, how exactly two cameras locate a point. Set the distance to the point and
  how precisely your own detector finds it; the panel gives the uncertainty sideways and in
  depth, and how far out the pair stays within 1 mm.

When the calibration was set up with a board that may be bowed, a line above the panel states
the bow Studio measured, from the lowest to the highest point of the board.

A photo that is blurred, has glare, or shows long lines all pointing one way can be left out of
the next run. Select it and choose **Leave this photo out next time**; its bar turns amber and a
bar under the timeline lists what you marked. **Run again without them** opens the setup with the
same settings and those photos listed under **Left out of this calibration**. The result you are
looking at does not change, and the new result names the photos that were left out of it.

Lengths are in the unit of your square size. Tell Studio once whether that was metres,
centimetres or millimetres and it shows millimetres.

/// note | A lower bound
The numbers under **What it means** count random error only, as the pair uncertainty does.
Real errors are usually larger.
///

**How close is this to what you need?** compares the weakest camera pair with the precision you
asked for when you set up the run, or that you type on the page. If it falls short, Studio estimates how many
photos of the same kind would close the gap. Treat that as rough: it assumes the new photos are
as varied as the ones you have. Photos with new angles can need far fewer.

Photos show when Studio can read them: anywhere on the computer running Studio, or, with `--only-roots` or
`--read-only`, under a folder given with `--root`. A result written by an
earlier version does not record which photo is which frame; Studio then matches them by file
number and says so.

### 4. Evidence for each camera

Open **All the numbers** for the charts below. Each view starts with its answer in one line.

- **Is there a pattern in the errors?**
    - A line chart of the error size from the image centre to its corners. A flat line is good;
      a rising one means the lens model does not fit near the edges.
    - The average error in each of 6 x 4 image regions. A full-length arrow equals the camera's
      typical error. Orange regions lean one way, which points at detector bias or a bent board.
    - **Show every corner** draws every corner's error.
- **Where can you trust this lens?** The image split into three zones, with the share of each:
    - *Trusted*: the board was seen there and the lens 1-sigma is within the limit.
    - *Never covered*: no photo had the board there, so the model is guessing.
    - *Uncertain*: the lens 1-sigma is over the limit.

    To widen the trusted zone, take photos with the board in the grey and orange areas.
    **Show uncertainty values** draws the 1-sigma per cell.
- **Worst photos**: the ten photos with the largest error, and how many of their corners are
  outside the inlier band. Retake or remove a photo far worse than the rest. If the same frame
  is worst in several cameras, check the camera sync.

The rest of the page shows the rig in 3D, the camera details and lens parameters, how well
each camera pair agrees, the calibration board's flatness, the photos the solver set aside, and
the technical details with links to every result file.

/// warning | Uncertainties are lower bounds
Every ± value comes from photo noise alone. On real captures the true error has been 2 to 3
times larger, so Studio writes them as "at least". To validate a calibration, take a second,
separate round of board photos and check the calibration against it. Studio labels every result
"not validated with a second capture" until that step exists.
///

## Older results

- Results written before the per-camera lens checks have no lens or split-check fields. Studio
  shows those values as "not recorded" instead of marking the camera as failed.
- Results written before the diagnostics files (`diagnostics/residuals.npz` and
  `diagnostics/lens_sigma.npz`) still get a verdict, but have no evidence views. Calibrate again
  with a current DS-MSP to get them.

## Build the interface from a source checkout

Released wheels include the built interface. In a git checkout, build it once:

```bash
cd web
npm ci
npm run build:studio      # writes ds_msp/studio/static
```

Without it, `ds-msp studio` serves a page that explains this, and the JSON API at
`/api/runs` still works.
