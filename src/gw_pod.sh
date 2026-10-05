#!/usr/bin/env bash
# Gaussian Wrapping on a rented GPU pod (the dry run of gw_dryrun.py, or a real garment).
#
#   bash gw_pod.sh setup                  # conda env + GW at 11e3b6f + its CUDA extensions (~20-40 min)
#   bash gw_pod.sh run <dataset> <out>    # train + extract (+ texture) at -r 1, logging time and memory
#   bash gw_pod.sh pack <out>             # <out>.tgz: the raw mesh and the logs, to copy back
#
# Pod: a 48 GB Ampere/Ada card (RTX A6000 or L40S) runs GW's pinned torch 2.3.1 / CUDA 12.1 as is;
# an image with the CUDA 12.1 toolkit (nvcc; a "devel" image), >= 32 GB of RAM (GW keeps the float32
# training images in host RAM: ~10 GB for 36 views at 24 MP) and ~50 GB of disk. Upload the dataset
# (gw_dryrun.py prepare writes gw_dataset_upload.tar) and this script; `tar xf` the dataset first.
# The run takes hours at 24 MP (GW's paper: 27 min at 1600 px on Tanks and Temples), so ask before
# starting one on a paid pod.
set -eo pipefail            # not -u: conda's activation scripts use unset variables
SHA=11e3b6fb5ca6f0a54e2b5587a09693488f3655af
WORK=${WORK:-$PWD}
GW=$WORK/GaussianWrapping
CUDA_VERSION=${CUDA_VERSION:-12.1}

conda_init() {
    if [ -x "$WORK/miniforge3/bin/conda" ]; then source "$WORK/miniforge3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null; then source "$(conda info --base)/etc/profile.d/conda.sh"
    else return 1; fi
}

cuda_env() {
    local root=${CUDA_ROOT:-/usr/local/cuda-$CUDA_VERSION}
    [ -d "$root" ] || root=/usr/local/cuda
    command -v "$root/bin/nvcc" >/dev/null || { echo "no nvcc under $root: use an image with the CUDA $CUDA_VERSION toolkit" >&2; exit 1; }
    "$root/bin/nvcc" --version | grep -q "release $CUDA_VERSION" || echo "warning: nvcc is not CUDA $CUDA_VERSION" >&2
    export CUDA_HOME=$root PATH=$root/bin:$PATH
    export CPATH=$root/targets/x86_64-linux/include:${CPATH:-}
    export LD_LIBRARY_PATH=$root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
}

setup() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    free -g | head -2
    df -h "$WORK" | tail -1
    local need=""
    [ -x /usr/bin/time ] || need="$need time"
    command -v git >/dev/null || need="$need git"
    if [ -n "$need" ]; then apt-get update -qq && apt-get install -y -qq $need; fi
    if ! conda_init; then
        curl -fsSL -o "$WORK/miniforge.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge3"
        conda_init
    fi
    [ -d "$GW" ] || git clone --recurse-submodules https://github.com/diego1401/GaussianWrapping "$GW"
    git -C "$GW" checkout "$SHA"
    git -C "$GW" submodule update --init --recursive
    cuda_env
    conda env list | grep -q "^gaussian_wrapping " || conda create -n gaussian_wrapping python=3.9 -y
    conda activate gaussian_wrapping
    (cd "$GW" && python install.py --cuda_version "$CUDA_VERSION") 2>&1 | tee "$WORK/gw_install.log"
    # install.py ignores failed builds (os.system), so check that GW's own entry points import
    (cd "$GW" && python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))")
    (cd "$GW" && python gaussian_wrapping/train.py --help >/dev/null && python gaussian_wrapping/pivot_based_mesh_extraction.py --help >/dev/null) \
        || { echo "GW does not import: see $WORK/gw_install.log" >&2; exit 1; }
    echo "setup ok"
}

run() {
    local ds=$1 out=$2
    [ -f "$ds/sparse/0/cameras.txt" ] || { echo "$ds is not a stage-1 GW dataset" >&2; exit 1; }
    conda_init; cuda_env; conda activate gaussian_wrapping
    mkdir -p "$out"
    export WANDB_MODE=disabled
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 5 > "$out/gpu.csv" 2>/dev/null &
    local smi=$!
    ( while sleep 30; do date +%s; free -m | sed -n 2p; done ) > "$out/ram.log" 2>&1 &
    local ram=$!
    (cd "$GW" && /usr/bin/time -v -o "$out/time.txt" \
        python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py -s "$(realpath "$ds")" -m "$(realpath "$out")" \
        --no_postprocess -r 1) 2>&1 | tee "$out/gw.log" || true
    kill $smi $ram 2>/dev/null || true
    wait $smi $ram 2>/dev/null || true          # the logs are complete, and nothing writes into $out any more
    ls -la "$out"/mesh_*searched*.ply
    awk -F', ' 'NF>=3 && $2>m {m=$2} END {printf "peak VRAM %.1f GB\n", m/1024}' "$out/gpu.csv"
    grep -E "Elapsed|Maximum resident" "$out/time.txt"
}

pack() {
    local out=$1
    tar czf "$out.tgz" -C "$(dirname "$out")" \
        $(cd "$(dirname "$out")" && ls "$(basename "$out")"/mesh_exact_computation_2pivots*_searched.ply \
          "$(basename "$out")"/{gpu.csv,ram.log,time.txt,gw.log} 2>/dev/null)
    ls -la "$out.tgz"
}

case "${1:-}" in
    setup) setup ;;
    run) run "$2" "$3" ;;
    pack) pack "$2" ;;
    *) sed -n 2,14p "$0"; exit 1 ;;
esac
