# Sourced by every reproduce/*.sh script. Portable: no cluster paths, no Slurm, run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV=${ENV:-$ROOT/envs/fidip}                 # conda prefix env built by 00_env.sh
PY=${PY:-$ENV/bin/python}                    # override with PY=/path/to/python to use another env
OUT=${OUT:-repro}                            # fresh outputs go here; the tracked results in output*/ are never touched
CFG_HRNET=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
CFG_MOBILE=experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml
INIT_HRNET=models/coco/w48_384x288.pth       # DarkPose COCO weights (in checkpoints/fidip_checkpoints.zip)
INIT_MOBILE=models/coco/posemobile.pth
export PYTHONDONTWRITEBYTECODE=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing: $f  (run the earlier reproduce/ steps first)" >&2; exit 1; }; done; }
[ -x "$PY" ] || { echo "python not found at $PY: run reproduce/00_env.sh or set PY=" >&2; exit 1; }

# arch -> config / eval output subdir
cfg_of()  { case "$1" in mobile) echo "$CFG_MOBILE";; *) echo "$CFG_HRNET";; esac; }

# official evaluation (tools/test_adaptive_model.py: GT boxes, flip test, COCO OKS AP)
#   evaluate <arch> <checkpoint> <data root> <output dir>
evaluate() {
  local arch=$1 ckpt=$2 root=$3 out=$4
  need "$ckpt" "$root"
  mkdir -p "$(dirname "$out")"
  "$PY" tools/test_adaptive_model.py --cfg "$(cfg_of "$arch")" \
    TEST.MODEL_FILE "$ckpt" TEST.USE_GT_BBOX True WORKERS "${WORKERS:-4}" DATASET.ROOT "$root" OUTPUT_DIR "$out" LOG_DIR "$OUT/log_eval" \
    | grep -E '^\| (Arch|---|adaptive)' || true
  find "$out" -name feature.npy -delete     # 0.6 GB debug dump written by every eval, never used
}
