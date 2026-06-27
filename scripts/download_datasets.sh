#!/usr/bin/env bash
# Download the free, small (<10 GB total) public datasets used by this project's
# examples and roadmap. All are resumable (curl -C -); safe to re-run.
#
#   bash scripts/download_datasets.sh            # download everything below
#   bash scripts/download_datasets.sh tumvi      # only the TUM-VI group
#   bash scripts/download_datasets.sh euroc       # EuRoC: MANUAL download, this only extracts it
#   bash scripts/download_datasets.sh tumrgbd     # only TUM RGB-D
#
# Files land under ./datasets/<group>/ (git-ignored). Archives are extracted
# in place and the tarball kept (delete by hand to reclaim space).
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="datasets"

get () {  # get <url> <out_relpath>
  local url="$1" out="$DEST/$2"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then echo "  ✓ have $out"; fi
  echo "  ↓ $url"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$out" "$url" \
    || { echo "  ✗ FAILED: $url"; return 1; }
}

extract () {  # extract <archive_relpath>
  local a="$DEST/$1"
  case "$a" in
    *.tar)      tar -xf "$a"   -C "$(dirname "$a")" ;;
    *.tgz|*.tar.gz) tar -xzf "$a" -C "$(dirname "$a")" ;;
    *.zip)      unzip -n -q "$a" -d "${a%.zip}" ;;
  esac
}

dl_tumvi () {
  echo "== TUM-VI (fisheye stereo + IMU + mocap GT + published Double Sphere calib) =="
  local base="https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16"
  get "$base/dataset-room1_512_16.tar"      tumvi/dataset-room1_512_16.tar      && extract tumvi/dataset-room1_512_16.tar
  get "$base/dataset-calib-cam1_512_16.tar" tumvi/dataset-calib-cam1_512_16.tar && extract tumvi/dataset-calib-cam1_512_16.tar
  get "$base/dataset-calib-imu1_512_16.tar" tumvi/dataset-calib-imu1_512_16.tar && extract tumvi/dataset-calib-imu1_512_16.tar
}

dl_tumrgbd () {
  echo "== TUM RGB-D (RGB+depth+GT pose; for learned-depth metric validation) =="
  get "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz" \
      tumrgbd/rgbd_dataset_freiburg1_xyz.tgz && extract tumrgbd/rgbd_dataset_freiburg1_xyz.tgz
}

dl_euroc () {
  echo "== EuRoC MAV — Vicon Room 1 (stereo + IMU + GT + radtan calib) =="
  # HONEST NOTE: EuRoC CANNOT be downloaded by this (or any) script from its current
  # host. The legacy direct host (robotics.ethz.ch/~asl-datasets) is OFFLINE, and the
  # current host (ETH Research Collection) serves a browser/JS bundle with no stable
  # curl-able URL. So this step does NOT download — it only EXTRACTS a bundle you have
  # already downloaded in a browser and placed under datasets/euroc/.
  if ls "$DEST"/euroc/*/mav0 >/dev/null 2>&1; then
    echo "  ✓ EuRoC sequences already extracted under datasets/euroc/ — nothing to do."
    return 0
  fi
  local z found=""
  for z in "$DEST"/euroc/vicon_room1.zip "$DEST"/euroc/*.zip; do
    [ -f "$z" ] && { found="$z"; break; }
  done
  if [ -n "$found" ]; then
    echo "  ↪ extracting $found (this bundle nests V1_0x*.zip inside)"
    unzip -n -q "$found" -d "$DEST/euroc"
    # the outer bundle contains per-sequence .zip (ASL) and .bag (ROS) — unpack the
    # ASL zips, ignore the bags
    for z in "$DEST"/euroc/*/*.zip "$DEST"/euroc/*.zip; do
      [ -f "$z" ] && unzip -n -q "$z" -d "$(dirname "$z")/$(basename "${z%.zip}")" 2>/dev/null
    done
    echo "  done. Verify with: ls datasets/euroc/*/mav0"
    return 0
  fi
  cat <<'EOF'
  EuRoC is a MANUAL download — no script can fetch it from the current host:
    1. Open the EuRoC MAV page on the ETH Research Collection
       (search: "EuRoC MAV dataset research-collection.ethz.ch").
    2. Download the "Vicon Room 1 Datasets" ZIP in your browser (~5.8 GB).
    3. Move it to:  datasets/euroc/vicon_room1.zip
    4. Re-run:      bash scripts/download_datasets.sh euroc   (this unpacks it)
  If you already have datasets/euroc/V1_0x_*/mav0/, you're done — ignore this.
EOF
}

case "${1:-all}" in
  tumvi)   dl_tumvi ;;
  tumrgbd) dl_tumrgbd ;;
  euroc)   dl_euroc ;;
  all)     dl_tumvi; dl_tumrgbd; dl_euroc ;;
  *) echo "usage: $0 [all|tumvi|tumrgbd|euroc]"; exit 1 ;;
esac
echo "Done. Total size:"; du -sh "$DEST" 2>/dev/null
