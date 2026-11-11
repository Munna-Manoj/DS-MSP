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

## Start Studio

```bash
ds-msp studio
```

```text
DS-MSP Studio: http://127.0.0.1:8765/?token=...
Press Ctrl+C to stop.
```

Studio opens the link in your browser. The link holds an access key, so use it as printed.

- **Where it looks.** Studio can read any folder on the computer it runs on. If you reach it
  from another device, its file choosers still show the computer running Studio.
- **Where results go.** New results are saved in the folder you started Studio from, each in its
  own folder. `--root DIR` changes that folder, and **Save somewhere else** on the last step picks
  another one.
- **Reading less.** With `--only-roots`, Studio reads only inside the folders given with `--root`.

## Set up the calibration

Choose **New calibration**. The page first asks only what you have. After you choose, a tracker on
the left shows your start, the six steps under it and how many required steps remain. One step
is open at a time: move with **Back** and **Next**, or click any step in the tracker. The last
step, **Name and start**, lists what is still missing, and each item leads to the step that fixes
it.

| You have | Use it when | What Studio does |
|---|---|---|
| **My photos** | You have a new capture | Reads the cameras and frames from the folder |
| **A settings file** | You have a `calib_param.yml`, from DS-MSP or MC-Calib | Fills in the steps from it |
| **Corners found earlier** | An earlier run wrote `detected_keypoints_data.yml` | Uses those corners; the photos are not read again |
| **A finished result** | You want the same calibration again, or one change | Offers **Repeat exactly** and **Change something first** per result |

For a settings file, **Choose a file where Studio runs** is best: Studio can then find the photos
and lens file the settings point to. **Upload a copy from this device**, or dropping the file on
the page, is for a file only on the device you are using. Studio then gets only its text, so you
choose again any path it cannot find.

On the page, a white box with a firm outline is something you fill in, grey text explains, and a
tinted panel with a mark is something Studio worked out.

The steps below describe starting from photos.

1. **Your photos.** Choose the folder that holds the camera folders (`Images` above). Studio
   shows one photo of each camera, the photo count, and the number of moments seen by two or
   more cameras. A folder layout or file naming it cannot use is named here, before anything
   runs.

   If your folders are not named `Cam_001`, `Cam_002` and so on, each appears as a card with one
   of its photos and the name it will go by. Set the order, because lens values go with the
   camera numbers. Mark any folder that is not a camera, then choose **Use these N cameras** (N is your camera count).
   Studio links your folders under the new names and does not rename them.

   After you choose this folder, every other file chooser on the page opens there. You can still
   browse elsewhere.
2. **Your calibration board.**
   - **Squares.** Choose **Find the board in my photos**: Studio reads the markers in about 48
     photos and fills in the squares across and down. When no photo shows a whole board, it says
     so, and you count the squares yourself.
   - **Size.** Enter the size of one square and its unit; the drawing marks the square to
     measure. **Look for this board** then draws every corner it finds in a few photos per camera.
   - **Marker size.** The same button measures a marker against its square; most boards are
     printed at 75 %. Another value is offered, never set by itself. A wrong value still finds the
     corners, but off their true place.
   - **Set the marker size yourself** under **My board was not printed with the usual settings**:
     drag the slider until the close-up looks like your board, or type one marker's width.
   - **Several boards.** Set how many are fixed together. If they differ, switch off **All boards
     are printed the same**, and each board gets its own row.
   - **Board number.** Boards printed as one set are numbered 0, 1, 2 and so on. Give yours under
     **My board was not printed with the usual settings**. Three of the seven MC-Calib sequences
     use board 12 or 13; with the default number, no board is found.
   - **Flatness.** Choose *It may be slightly bowed* for a board on paper, foam or thin plastic;
     Studio then measures the bow. In simulation, a 2.6 mm bow treated as flat put the focal
     length 4.7 % off; with this choice, 0.3 %. A flat board reports about 0.1 mm from corner
     noise. Several boards get four choices, from stiff boards to every corner free.
3. **Your cameras and their lenses.** Each camera has a card with one of its photos.
   **Measure my lenses** measures how wide each lens sees and sets its type: a standard lens
   under 100° across, *Fisheye* at 100° or wider. You can change any card afterwards.

   /// warning | Set the lens type yourself when you know it
   The measurement picks *Fisheye* for every lens at 100° or wider, including a standard lens
   at exactly 100° and EUCM, UCM or Double Sphere lenses. On five rendered 25-camera rigs,
   keeping those picks moved the worst camera by 1.3 to 10.0 mm, against 0.20 to 0.43 mm with
   the true types. The result page still read 25 of 25 cameras ready, because the pixel error
   stayed low.
   ///

   Under **Where do the lens values come from?** choose one of three:
   - **Work them out**: Studio computes them from these photos. Not available with strict
     checking, which keeps lens values exactly as you give them.
   - **From a lens file**: a file in the format of `calibrated_cameras_data.yml`, from MC-Calib
     or DS-MSP. Studio names the file it is using; a file it cannot use is named with what kind
     of file it is, right where you chose it.
   - **I'll type them in**: each card asks for the values of its lens type. If the type does not
     suit the photos, Studio warns you and still lets you enter the values you computed.

   With a file or typed values, say whether Studio should improve them or keep them exactly as
   they are. Keeping them needs values for every camera.
4. **What should happen with a doubtful result?** *Give it to me with warnings* always writes a
   result and lists its warnings. *Refuse it* writes no result unless every check passes.
   It needs the two limits below it: how precisely the angle and the position between two
   cameras must be known. The result page compares the result with those limits.
5. **More settings.** Optional; see below.
6. **Name and start.** Lists your choices, each linked to its step, and anything still missing.

Studio checks what you entered against the photos before anything runs:

- **The lens type.** Each camera's lens is measured on up to 12 photos and the chosen type is
  fitted to them. The card then reads, for example, `68° wide · this type fits (0.05 px)`. A
  type that cannot fit is marked, with a button that switches to one that can.
- **A lens file.** Each camera's given values are tried on its photos with the lens held
  fixed. Values for another camera, another image size or another lens setting show up as a
  large error, named per camera.
- **Earlier calibrations of the same photos.** When the photos and the board layout are
  unchanged, Studio offers **Skip finding the corners again**. The earlier result is copied
  from and never changed.

When you start from **Corners found earlier**, the lens checks use up to 12 of the saved board
views per camera, so the photos are not needed. If the board you entered is not the one the
corners were found with, the card says so.

/// note | What the lens check cannot see
It compares pixel errors. A lens type can fit the pixels and still bias the rig; the one known
case, UCM on an ordinary lens, is flagged by rule.
///

**More settings** holds everything else. A normal calibration needs none of it. Each setting is
a question with the reason it exists, when to change it, and its `calib_param.yml` name in small
type. A setting you do not touch is left out of the config file, so the default applies.

- **When is a found corner a mistake?** (`noise_bound`). You set the distance in pixels beyond
  which a corner is ignored. When the photos already have a result, the picture shows that
  result's corners, and the page says how far off its worst corner was and how many corners the
  setting would ignore. Without a result it shows an example and says so.
- **Checks before you get a result** (the `*_gate` keys). Each check is a question with **Skip**,
  **Warn me** or **Refuse the result**. With *Refuse it* chosen in step 4 they show the stricter
  values that choice applies.
- **Files for a device or a simulator** (`export_ldc`, `export_isaac_lut` and their settings).
  Writes a TI LDC mesh and an Isaac Sim lens LUT for each camera into the result folder. The
  result page then lists each mesh with its measured error, and marks one that must not be
  used. See [Export LDC meshes and Isaac Sim LUTs](export_lens_files_in_studio.md).
- Keys of a loaded config that this page shows another way (MC-Calib's `distortion_per_camera`,
  for example) are turned into the page's own choices.

/// tip | Start from an existing config
**A settings file** on the first page fills every step from a `calib_param.yml`, including
MC-Calib's own, and opens the first step that still needs something. On a result page,
**Change settings and run again** starts from the settings of that run.
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

## Run again without a bad photo

On the result page, select a photo that fits badly and choose **Leave this photo out next time**,
then **Run again without them**. The setup opens with the settings of that result and the marked
photos under, for example, **2 photos are left out of this calibration**; the × on an entry puts that photo back. When the
photos and the board are unchanged, Studio reuses the corners already found, so only the
calibration itself runs again. The new result goes to its own folder.

The list is the `exclude_views` key of `calib_param.yml`, so the same run works from a terminal:

```yaml
exclude_views: [ "0:7", "0:29" ]   # camera 0, frames 7 and 29
```

This also works for a result made from a terminal, when its `calib_param.yml` lies under a
folder given with `--root`.

## Repeat a calibration exactly

Every saved calibration records its settings, a checksum of every input file and the software
that ran it. On a result page choose **Repeat exactly**:

1. **What this run recorded** shows that record.
2. **Check the input files** reads each one again and compares checksums. A changed or missing
   file is listed by name, and the repeat is held back until it is restored.
3. **Repeat exactly** runs the recorded settings into a new folder. The original is not touched.

When the repeat ends, its page compares it with the original, file by file:

```text
Same calibration as the original
✓ corner detections used            identical
✓ target used                       identical
✓ calibrated_cameras_data.yml       identical
✓ calibrated_objects_pose_data.yml  identical
```

A result made from a scenario folder, without a `calib_param.yml`, cannot be repeated here;
use `tools/replay_rig_manifest.py` from a source checkout.

## Repeat the run from a terminal

The run page shows the command:

```bash
ds-msp-calibrate-rig --config path/to/my_rig/my_first_run/calib_param.yml
```

It reads the config Studio wrote and gives the same calibration. On the seven MC-Calib real
sequences and six rendered 25-camera rigs, the terminal run wrote byte-identical camera, target
and detection files.

## What Studio adds to a result folder

| File | Holds |
|---|---|
| `calib_param.yml` | The settings of the run, in the format the CLI reads |
| `studio_job.json` | When the run started and ended, and how it ended |
| `studio_run.log` | Everything the calibration printed |
| `studio_progress.jsonl` | The progress log the run page follows |
| `camera_lineup.json` | Which of your folders is which camera, if you set that on the page |
| `entered_lens_values.yml` | The lens values you typed in, if you typed them |
| `ldc/`, `isaac_lut/`, `exports.json` | The LDC meshes and Isaac LUTs, if you asked for them in **More settings** |

## View only

```bash
ds-msp studio --root path/to/results --read-only
```

With `--read-only`, Studio shows results and cannot set up or start a calibration.
