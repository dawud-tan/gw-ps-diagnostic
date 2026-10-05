# Gaussian Wrapping dry run on a rented GPU

What it answers before the rig exists: does GW at commit 11e3b6f install and run on our dataset format (PINHOLE cameras with the principal point centred by stage 1, RGBA images whose alpha is the object mask), how much GPU memory, host memory and time it needs at `-r 1`, how close its mesh comes to a known surface, and whether stages 2–7 and the bake run on a real GW mesh. Everything except the GPU step runs on this machine; the kit's own test (`tests/test_W_gw_dryrun.py`) runs the whole chain here with stand-ins for GW's mesh.

The synthetic garment is the fold-textured torso of `gwps.synth` on a textured turntable top, in front of a textured studio, seen through the distorted pilot camera, 36 turntable steps of 10° with 1° of jitter, with silhouette, SfM and single-light PS frames (`gwps/synth_sfm.py`).

## 1. Here: prepare (about 10 minutes at 1920 × 1440)

```bash
PYTHONPATH=src .venv/bin/python src/gw_dryrun.py prepare --out runs/gw_dryrun --bump
```

- `--bump` puts a 3 mm bump (σ 10 mm) on the garment. A GW mesh that keeps it should come out **BAKE**, one that smooths it away **BUILD** with the bump flagged. Without `--bump`, BAKE is the only right answer.
- `--scale 3.125` renders 6000 × 4500 (24 MP, the planned camera) instead, for GW's real memory and time: slower here, and GW then takes hours.
- It writes `runs/gw_dryrun/gw_dataset_upload.tar` (images and sparse model only; ~50 MB at 1920 × 1440).

## 2. Rent the pod and upload

- **GPU:** a 48 GB Ampere or Ada card (RTX A6000 or L40S) runs GW's pinned torch 2.3.1 with CUDA 12.1 as is; RTX 50-series cards would need a newer torch and a rebuild.
- **Image:** one with the **CUDA 12.1 toolkit** (`nvcc`; a "devel" image), because GW compiles its CUDA extensions.
- **Host:** at least 32 GB of RAM (GW keeps the training images in host memory) and ~50 GB of disk.
- Upload `gw_dataset_upload.tar` and `src/gw_pod.sh` (scp over the pod's SSH, or the provider's file transfer), then `tar xf gw_dataset_upload.tar`.
- **Ask before starting a long paid run** (CLAUDE.md working agreement: GPU runs over ~30 minutes).

## 3. On the pod

```bash
bash gw_pod.sh setup                    # Miniforge, GW at 11e3b6f, its CUDA extensions; checks that GW imports
bash gw_pod.sh run gw_dataset gw_out    # train + extract + texture at -r 1, --no_postprocess
bash gw_pod.sh pack gw_out              # gw_out.tgz: the raw mesh, gpu.csv, ram.log, time.txt, gw.log
```

`run` logs GPU memory every 5 s (`gpu.csv`), host memory every 30 s (`ram.log`), and wall time and peak memory (`time.txt`), and prints the peak VRAM at the end. Copy `gw_out.tgz` back and stop the pod.

## 4. Here: finish

```bash
tar xzf gw_out.tgz -C runs/gw_dryrun/
PYTHONPATH=src .venv/bin/python src/gw_dryrun.py finish --work runs/gw_dryrun \
    --mesh runs/gw_dryrun/gw_out/mesh_exact_computation_2pivots_searched.ply --pod-logs runs/gw_dryrun/gw_out
```

It runs stage 2 (frame gates against the GW mesh), stage 3, garment selection (here the region cut: the torso's side between z = 0.02 and 0.58 m), stages 4, 5 and 7, and the bake. It then writes `runs/gw_dryrun/finish/finish.md` with:
- the verdict and its clusters;
- the GW mesh's signed distance to the true surface (median, p95, max, mean);
- the bake;
- the pod's peak VRAM, wall time and peak host RAM.

## What to record in CLAUDE.md afterwards

- **The install:** it worked or not, and on which GPU and image.
- **The pod's numbers:** peak VRAM, host RAM and wall time.
- **GW on synthetic data:** its distance to the true surface.
- **The verdict:** whether GW kept the 3 mm bump. This is a first synthetic look at the BAKE / BUILD question, not the real answer: real fabric, lighting and GW behaviour will differ.
