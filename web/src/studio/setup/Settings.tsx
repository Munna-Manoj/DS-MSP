// "More settings": everything a calibration can be told beyond photos, board and lenses. Each
// setting is a question in plain words, with why it exists and when to change it, and a control
// that fits its value. Nothing here has to be touched for a normal calibration.
import type { ReactNode } from "react";

import type { ConfigSchema, Problem, Values } from "../api";
import { OutlierBand, TargetShape } from "../Teach";
import { Button, INPUT } from "../ui";
import { Explain, KeyTag, Mono, NumberBox, Range, Segmented, Toggle } from "./kit";

type Set = (key: string, v: unknown) => void;

// Settings with their own place earlier on the page, or written by Studio itself.
const ELSEWHERE = new Set([
  "root_path", "cam_prefix", "number_camera", "camera_models", "number_x_square", "number_y_square",
  "square_size", "number_board", "profile", "max_rotation_sigma_deg", "max_centre_sigma", "save_path",
  "webviewer", "keypoints_path", "cam_params_path", "fix_intrinsic", "exclude_views", "length_square",
  "length_marker", "boards_index", "number_x_square_per_board", "number_y_square_per_board",
  "square_size_per_board",
]);
const HERE = new Set([
  "noise_bound", "reject_outliers", "ransac_threshold", "structure_mode", "structure_prior_sigma",
  "number_iterations", "he_approach", "audit_gate", "uncertainty_gate", "convergence_gate", "coverage_gate",
  "min_camera_coverage", "lens_gate", "max_lens_sigma_px", "crossval_gate", "accuracy_gate",
  "accuracy_profile_path", "save_detection", "save_reprojection", "report_covariance", "certify", "verbose",
  "camera_params_file_name", "frame_pattern", "object_path", "keypoints_pixel_convention",
]);

/** One setting: the question, why it is there, and its control. */
function Setting({ title, k, changed, onReset, why, when, problem, wide, children }: {
  title: string; k: string; changed: boolean; onReset: () => void; why: ReactNode; when?: ReactNode;
  problem?: string | null; wide?: boolean; children: ReactNode;
}) {
  return (
    <div className={`py-5 first:pt-1 grid gap-x-8 gap-y-3 items-start ${wide ? "" : "lg:grid-cols-[minmax(0,1fr)_minmax(260px,340px)]"}`}>
      <div className="grid gap-1.5 min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h4 className="text-[17px] font-semibold">{title}</h4>
          {changed && <span className="rounded-full bg-s-accent text-white text-[13px] font-semibold px-2 py-0.5">changed</span>}
          {changed && <button type="button" onClick={onReset} className="text-[15px] text-s-accent underline cursor-pointer">Back to the usual value</button>}
        </div>
        <Explain>{why}</Explain>
        {when && <Explain><b className="font-semibold text-s-text">Change it when </b>{when}</Explain>}
        <KeyTag name={k} />
      </div>
      <div className="grid gap-2 min-w-0 content-start">
        {children}
        {problem && <div className="text-[15px] font-medium text-s-bad" role="alert">{problem}</div>}
      </div>
    </div>
  );
}

function Group({ title, intro, children }: { title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <section className="grid gap-1">
      <h3 className="text-lg font-bold tracking-tight border-b-2 border-s-text/80 pb-1.5">{title}</h3>
      {intro && <Explain className="pt-2">{intro}</Explain>}
      <div className="divide-y divide-s-line">{children}</div>
    </section>
  );
}

const GATE_OPTIONS = [
  { id: "off", title: "Skip" },
  { id: "warn", title: "Warn me", tone: "warn" as const },
  { id: "refuse", title: "Refuse the result", tone: "bad" as const },
];
const GATE_MEANS: Record<string, string> = {
  off: "This check is not made.",
  warn: "You get the result, with a warning if this check fails.",
  refuse: "No result is written if this check fails.",
};

export function Settings({ schema, values, problems, set, onBrowse, refRun, boards, source, strict }: {
  schema: ConfigSchema; values: Values; problems: Problem[]; set: Set; onBrowse: (key: string) => void;
  refRun: { id: string; name: string } | null; boards: number; source: string;
  /** Step 4 is set to refuse a doubtful result: untouched checks then take the strict defaults. */
  strict: boolean;
}) {
  const P = schema.properties;
  const usual = (strict ? schema["x-profiles"]?.production : undefined) ?? {};
  const has = (k: string) => values[k] !== undefined && values[k] !== "";
  const val = <T,>(k: string): T => (has(k) ? values[k] : k in usual ? usual[k] : P[k].default) as T;
  const reset = (k: string) => () => set(k, "");
  const problem = (k: string) => problems.find((p) => p.key === k)?.message ?? null;
  const shownKeys = Object.keys(P).filter((k) => HERE.has(k));
  const changed = shownKeys.filter(has);
  const bad = problems.filter((p) => HERE.has(p.key)).length;
  const others = Object.keys(values).filter((k) => !ELSEWHERE.has(k) && !HERE.has(k));

  const gate = (k: string, title: string, why: ReactNode, extra?: ReactNode) => (
    <Setting title={title} k={k} changed={has(k)} onReset={reset(k)} why={why} problem={problem(k)}>
      <Segmented label={title} value={String(val<string>(k))} options={GATE_OPTIONS.filter((o) => (P[k].enum ?? []).map(String).includes(o.id))}
        onChange={(v) => set(k, v)} />
      <span className="text-[15px] text-s-muted">{GATE_MEANS[String(val<string>(k))]}</span>
      {extra}
    </Setting>
  );
  const toggle = (k: string, title: string, why: ReactNode, on = "Yes", off = "No") => (
    <Setting title={title} k={k} changed={has(k)} onReset={reset(k)} why={why} problem={problem(k)}>
      <Toggle id={k} on={Boolean(val<boolean>(k))} onChange={(v) => set(k, v)}>{val<boolean>(k) ? on : off}</Toggle>
    </Setting>
  );
  const file = (k: string, label: string) => (
    <div className="flex gap-2">
      <input id={k} className={`${INPUT} font-studio-mono !text-sm`} readOnly placeholder="No file chosen" value={has(k) ? String(values[k]) : ""} />
      <Button onClick={() => onBrowse(k)} className="whitespace-nowrap">{label}</Button>
    </div>
  );

  return (
    <details className="rounded-2xl bg-s-surface border border-s-line shadow-[0_1px_2px_rgb(15_23_42/0.06),0_4px_16px_rgb(15_23_42/0.05)]" open={bad > 0 || undefined}>
      <summary className="cursor-pointer px-5 sm:px-6 py-5 flex flex-wrap items-center gap-3 hover:bg-s-sunken rounded-2xl">
        <span className="size-9 shrink-0 rounded-full grid place-items-center text-base font-bold bg-s-sunken text-s-muted border border-s-line" aria-hidden="true">5</span>
        <h2 className="text-xl font-semibold tracking-tight">More settings</h2>
        <span className="rounded-full bg-s-sunken text-s-muted text-[13px] font-medium px-2.5 py-0.5">optional</span>
        <span className="text-[15px] text-s-muted">A normal calibration needs none of these.</span>
        {changed.length > 0 && <span className="rounded-full bg-s-accent text-white text-[13px] font-semibold px-2.5 py-0.5">{changed.length} changed</span>}
        {bad > 0 && <span className="rounded-full bg-s-bad text-white text-[13px] font-semibold px-2.5 py-0.5">{bad} to fix</span>}
      </summary>
      <div className="px-5 sm:px-6 pb-6 grid gap-9">

        <Group title="Mistakes in the photos">
          <Setting wide title="When is a found corner a mistake?" k="noise_bound" changed={has("noise_bound")} onReset={reset("noise_bound")} problem={problem("noise_bound")}
            why={<>Sometimes a corner is found in the wrong place: glare on the board, motion blur, a marker read as another one. A few such corners can pull the whole calibration off. So a corner that lies far from where all the other corners say it should be is treated as a mistake and left out.</>}
            when={<>a result shows clearly wrong corners that were still used (make it stricter), or warns that many good corners were ignored (make it more lenient).</>}>
            <OutlierBand value={has("noise_bound") ? Number(values.noise_bound) : undefined} fallback={Number(P.noise_bound.default)}
              refRun={refRun} onChange={(v) => set("noise_bound", v)} />
          </Setting>
          <Setting title="Remove whole views of the board that fit badly" k="reject_outliers" changed={has("reject_outliers") || has("ransac_threshold")}
            onReset={() => { set("reject_outliers", ""); set("ransac_threshold", ""); }} problem={problem("reject_outliers") ?? problem("ransac_threshold")}
            why={<>Besides single corners, a whole view can be wrong, for example when the board moved during the exposure. With this on, a view whose corners are off by more than the limit is removed before the final calibration.</>}
            when={<>a result shows whole photos that are clearly wrong. For one or two known bad photos, mark them on the result page instead.</>}>
            <Toggle id="reject_outliers" on={Boolean(val<boolean>("reject_outliers"))} onChange={(v) => set("reject_outliers", v)}>
              {val<boolean>("reject_outliers") ? "Remove them" : "Keep every view"}
            </Toggle>
            {Boolean(val<boolean>("reject_outliers")) && (
              <label className="grid gap-1 text-[15px] font-medium">
                when its corners are off by more than
                <NumberBox id="ransac_threshold" value={values.ransac_threshold} fallback={Number(P.ransac_threshold.default)} unit="px" onChange={(v) => set("ransac_threshold", v)} />
              </label>
            )}
          </Setting>
        </Group>

        {boards > 1 && (
          <Group title={`Your target of ${boards} boards`}>
            <Setting wide title="How well do you know the shape of the target?" k="structure_mode" changed={has("structure_mode")} onReset={reset("structure_mode")} problem={problem("structure_mode")}
              why={<>Studio first estimates how the boards are mounted relative to one another, then can improve that together with the cameras. How much it may change depends on how stiff and flat your target really is.</>}
              when={<>your boards are not rigid (choose bowed), or a result reports that the boards do not fit their printed shape.</>}>
              <TargetShape value={has("structure_mode") ? String(values.structure_mode) : undefined} fallback={String(P.structure_mode.default)} onChange={(v) => set("structure_mode", v)} />
            </Setting>
            {val<string>("structure_mode") === "free" && (
              <Setting title="How far may a corner move from its printed place?" k="structure_prior_sigma" changed={has("structure_prior_sigma")} onReset={reset("structure_prior_sigma")} problem={problem("structure_prior_sigma")}
                why={<>With every corner free, this keeps each one near where the print says it is. Without a limit the target can absorb errors that belong to the lens.</>}>
                <NumberBox id="structure_prior_sigma" value={values.structure_prior_sigma} fallback={null} unit="units" onChange={(v) => set("structure_prior_sigma", v)} />
                <span className="text-[15px] text-s-muted">In the unit of your square size. A typical print is flat to about a thousandth of the board width.</span>
              </Setting>
            )}
          </Group>
        )}

        <Group title="Checks before you get a result" intro={<>Each check can be skipped, give a warning, or stop the result from being written. {strict ? "You chose in step 4 to refuse a doubtful result, so most of them refuse." : "You chose in step 4 to get a doubtful result with warnings, so they warn."} Change single ones here.</>}>
          {gate("convergence_gate", "Did the calibration finish properly?", <>The last stage improves the result round by round. It should stop because nothing improves any more, not because it ran out of rounds.</>)}
          {gate("coverage_gate", "Does the result explain most corners of every camera?", <>If many corners of one camera had to be ignored, that camera probably has the wrong lens type or bad photos.</>,
            String(val<string>("coverage_gate")) !== "off" && (
              <Range id="min_camera_coverage" value={Math.round(100 * Number(val<number>("min_camera_coverage")))} min={50} max={100} step={1} unit="%"
                words="of each camera's corners must be used" onChange={(v) => set("min_camera_coverage", v / 100)} />
            ))}
          {gate("audit_gate", "Is every camera really pinned down?", <>Looks for a direction in which a camera could shift or turn without the photos showing it. That happens when the board was only ever seen in one spot or at one angle.</>)}
          {gate("uncertainty_gate", "Are the cameras known as precisely as you need?", <>Compares how uncertain each pair of cameras is with the two limits you give in step 4. Without limits there is nothing to compare.</>)}
          {gate("lens_gate", "Is each lens known precisely enough?", <>Checks how uncertain each lens is, in pixels, across its image, and whether one of its values ran into a limit.</>,
            String(val<string>("lens_gate")) !== "off" && (
              <label className="grid gap-1 text-[15px] font-medium">
                A lens may be uncertain by at most
                <NumberBox id="max_lens_sigma_px" value={values.max_lens_sigma_px} fallback={Number(has("noise_bound") ? values.noise_bound : P.noise_bound.default)} unit="px" onChange={(v) => set("max_lens_sigma_px", v)} />
              </label>
            ))}
          {gate("crossval_gate", "Do two halves of the photos agree about each lens?", <>Calibrates every lens twice, on alternating photos, and compares the two. It takes longer, so it is off unless you ask for it.</>)}
          <Setting title="Compare with measurements you made yourself" k="accuracy_gate" changed={has("accuracy_gate") || has("accuracy_profile_path")}
            onReset={() => { set("accuracy_gate", ""); set("accuracy_profile_path", ""); }} problem={problem("accuracy_gate") ?? problem("accuracy_profile_path")}
            why={<>If you measured the rig independently, an accuracy profile file holds those measurements and how far the calibration may differ from them. With this on, a result that differs more is refused.</>}>
            <Toggle id="accuracy_gate" on={val<string>("accuracy_gate") === "refuse"} onChange={(v) => set("accuracy_gate", v ? "refuse" : "")}>
              {val<string>("accuracy_gate") === "refuse" ? "Refuse a result that disagrees" : "Not compared"}
            </Toggle>
            {file("accuracy_profile_path", "Choose")}
          </Setting>
        </Group>

        <Group title="How the calibration runs">
          <Setting title="How many rounds may the last stage take?" k="number_iterations" changed={has("number_iterations")} onReset={reset("number_iterations")} problem={problem("number_iterations")}
            why={<>The last stage improves the result in rounds and usually stops by itself long before this number.</>}
            when={<>a result says the calibration did not finish. Raise it then.</>}>
            <Range id="number_iterations" value={Number(val<number>("number_iterations"))} min={100} max={5000} step={100} unit="rounds at most" onChange={(v) => set("number_iterations", v)} />
          </Setting>
          <Setting title="Placing cameras that never see the board together" k="he_approach" changed={has("he_approach")} onReset={reset("he_approach")} problem={problem("he_approach")}
            why={<>Two cameras that never see the board at the same moment are linked through how the board moved. There are two ways to get the first estimate; both are improved afterwards.</>}
            when={<>a result says a camera could not be linked to the others. Try the other method.</>}>
            <Segmented label="Linking method" value={String(val<number>("he_approach"))}
              options={[{ id: "0", title: "Bootstrapped" }, { id: "1", title: "Classic" }]} onChange={(v) => set("he_approach", Number(v))} />
          </Setting>
        </Group>

        <Group title="Extra things to save">
          {toggle("save_detection", "A copy of every photo with the found corners drawn on it", <>Useful to show someone what was detected. Needs about as much disk space as the photos themselves.</>, "Saved", "Not saved")}
          {toggle("save_reprojection", "A copy of every photo with found and expected corners", <>Shows on each photo where the calibration expects every corner next to where it was found. The result page shows the same without the extra files.</>, "Saved", "Not saved")}
          {toggle("report_covariance", "The full table of uncertainties", <>For software that needs to know how every value of the calibration varies together with every other. People rarely read it.</>, "Saved", "Not saved")}
          {toggle("certify", "Proof that the camera rotations are the best possible", <>Checks mathematically that the rotations found are the best fit overall, not merely a good one nearby. Takes extra time.</>, "Computed", "Not computed")}
          {toggle("verbose", "Progress details in the log", <>What each stage did, written to the log you see while the calibration runs.</>, "Detailed", "Brief")}
          <Setting title="Name of the lens file that is written" k="camera_params_file_name" changed={has("camera_params_file_name")} onReset={reset("camera_params_file_name")} problem={problem("camera_params_file_name")}
            why={<>The result folder holds one file with every camera's lens values and position. Other tools look for it under the usual name, <Mono>calibrated_cameras_data.yml</Mono>.</>}>
            <input id="camera_params_file_name" className={`${INPUT} font-studio-mono !text-sm`} placeholder="calibrated_cameras_data.yml"
              value={has("camera_params_file_name") ? String(values.camera_params_file_name) : ""} onChange={(e) => set("camera_params_file_name", e.target.value)} />
          </Setting>
        </Group>

        <Group title="File names and earlier work">
          <Setting title="Which number in a file name says when the photo was taken?" k="frame_pattern" changed={has("frame_pattern")} onReset={reset("frame_pattern")} problem={problem("frame_pattern")}
            why={<>Photos taken at the same moment by different cameras must carry the same number. Studio uses the number in the file name, which works for names like <Mono>0042.png</Mono>.</>}
            when={<>your file names hold several numbers, such as <Mono>cam2_run7_0042.png</Mono>. Give a pattern whose part named <Mono>frame</Mono> picks the right one.</>}>
            <input id="frame_pattern" className={`${INPUT} font-studio-mono !text-sm`} placeholder="run\d+_(?P<frame>\d+)"
              value={has("frame_pattern") ? String(values.frame_pattern) : ""} onChange={(e) => set("frame_pattern", e.target.value)} />
          </Setting>
          <Setting title="Use a target that was already worked out" k="object_path" changed={has("object_path")} onReset={reset("object_path")} problem={problem("object_path")}
            why={<>Normally Studio works out from the photos how the boards of your target are mounted. If an earlier calibration already did that for the same target, its <Mono>calibrated_objects_data.yml</Mono> can be used as it is.</>}>
            {file("object_path", "Choose")}
          </Setting>
          {source === "detections" && (
            <Setting title="How were the saved corners measured?" k="keypoints_pixel_convention" changed={has("keypoints_pixel_convention")} onReset={reset("keypoints_pixel_convention")} problem={problem("keypoints_pixel_convention")}
              why={<>Files written by old DS-MSP versions on OpenCV 4 hold corners half a pixel off, and say nothing about it. Newer files state how they were measured.</>}
              when={<>the file comes from such an old version and Studio warns that it carries no label.</>}>
              <Segmented label="Pixel convention" value={has("keypoints_pixel_convention") ? String(values.keypoints_pixel_convention) : ""}
                options={[{ id: "", title: "As the file says" }, { id: "opencv4_charuco", title: "Old file, half a pixel off" }]}
                onChange={(v) => set("keypoints_pixel_convention", v)} />
            </Setting>
          )}
        </Group>

        {others.length > 0 && (
          <Group title="Other entries from your config file" intro={<>These came with the file you loaded. Studio writes the choices you made on this page in their place where both say the same thing; you can remove any of them.</>}>
            <ul className="flex flex-wrap gap-2 pt-3">
              {others.map((k) => (
                <li key={k} className="inline-flex items-center gap-2 rounded-full bg-s-sunken border border-s-line pl-3.5 pr-1 min-h-10 text-[15px]">
                  <span className="font-studio-mono text-sm">{k}: {Array.isArray(values[k]) ? `[${(values[k] as unknown[]).join(", ")}]` : String(values[k])}</span>
                  <button type="button" aria-label={`Remove ${k}`} onClick={() => set(k, "")} className="size-8 grid place-items-center rounded-full hover:bg-s-line text-s-muted cursor-pointer text-lg">×</button>
                </li>
              ))}
            </ul>
          </Group>
        )}
      </div>
    </details>
  );
}
