# Calibrate a rig in DS-MSP Studio

Set up a rig calibration in your browser, start it, watch it run and open its result, without
writing a config file. Studio runs the same calibration as `ds-msp-calibrate-rig` and saves the
config it used, so the run can be repeated from a terminal.

## Before you start

- Install the interface: `pip install "ds-msp[ui]"`.
- Put each camera's photos in its own folder: `Cam_001`, `Cam_002` and so on, numbered from 001
  with no gaps.
- Photos taken at the same moment must carry the same number in their file name, such as
  `0007.png` in every camera folder.
- Measure one printed square of your ChArUco board.

```text
my_rig/
  Images/
    Cam_001/0000.png 0001.png ...
    Cam_002/0000.png 0001.png ...
```

## Start Studio on the folder that holds the photos

```bash
ds-msp studio --root path/to/my_rig
```

```text
DS-MSP Studio: http://127.0.0.1:8765/
Press Ctrl+C to stop.
```

Studio reads and writes only inside the folders given with `--root`. Results are saved there
too, each in its own folder.

## Set up the calibration

Choose **New calibration**. The page has four steps and a Start panel.

1. **Photos.** Choose the folder that holds the camera folders (`Images` above). Studio reports
   the cameras it found, the photo count, and the number of moments seen by two or more cameras.
   A folder layout or file naming it cannot use is named here, before anything runs.
2. **Calibration board.** Enter the squares across, the squares down and the size of one square.
   The size sets the unit of the result: enter metres to get metres. **Look for this board in
   my photos** tries a few photos per camera and says in how many it found the board.
3. **Cameras.** Pick a lens type per camera. Use *Standard lens* for ordinary lenses and
   *Fisheye (KB)* for wide ones.
4. **How strict should the checks be?** *Standard* always writes a result and lists its
   warnings. *Production* refuses a result that fails a check and needs two uncertainty limits.

**More settings** lists every other `calib_param.yml` key with its default. A setting you do
not touch is left out of the config file, so the default applies.

/// tip | Start from an existing config
**Load a calib_param.yml** fills the form from a file, including MC-Calib's own. On a result
page, **New calibration with these settings** starts from the settings of that run.
///

## Run it

Name the calibration and choose **Start calibration**. Studio creates a folder with that name,
writes `calib_param.yml` into it and starts the calibration.

The run page shows:

- what the calibration is doing now, in plain words, and each finished stage with its time
- per-camera progress while it looks for the board in the photos
- the typical error while it solves
- everything the calibration printed

A rendered three-camera capture of 90 photos at 800 x 600 finishes in 3 s on a desktop computer.

**Cancel** stops the run. No result is written, and the settings stay in the folder.

Calibrations run one at a time, in the order they were started. Each one uses every processor
core, so two at once would only slow each other down.

## Open the result

When the run ends, choose **Open the result**. The result page is the one described in
[View rig results in DS-MSP Studio](view_rig_results_in_studio.md).

If the run stops without a result, the page shows the reason at the top and the full output
below it. A board description that does not match the target reads:

```text
The calibration board was not found in any image under /path/to/my_rig/Images ...
Check the board description (number_x_square x number_y_square = 9 x 7 squares, 1 board(s)) ...
```

## Repeat the run from a terminal

The run page shows the command:

```bash
ds-msp-calibrate-rig --config path/to/my_rig/my_first_run/calib_param.yml
```

It reads the config Studio wrote and gives the same calibration.

## What Studio adds to a result folder

| File | Holds |
|---|---|
| `calib_param.yml` | The settings of the run, in the format the CLI reads |
| `studio_job.json` | When the run started and ended, and how it ended |
| `studio_run.log` | Everything the calibration printed |
| `studio_progress.jsonl` | The progress log the run page follows |

## View only

```bash
ds-msp studio --root path/to/results --read-only
```

With `--read-only`, Studio shows results and cannot set up or start a calibration.
