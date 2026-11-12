# Future Massive Black Hole Mergers

A CNN that detects inspiralling massive black hole binaries (MBHBs) in LISA TDI A/E
spectrograms, following Houba et al., [arXiv:2405.11340](https://arxiv.org/abs/2405.11340).
It extends the paper's single-channel (A) network to take A and E stacked as channels.

## Layout
- `src/detection_cnn.py` – the CNN (Keras); width/depth/dropout are arguments
- `src/simulate.py` – IMRPhenomD + LISA response waveforms via `bbhx` (CPU or GPU)
- `src/spectrograms.py` – time series -> fixed-size log-power spectrogram images (batched, CPU or GPU)
- `src/dataset.py` – generates a dataset as a directory of shards
- `src/shards.py` – on-disk dataset format and reader
- `src/train_compare.py` – trains A-only vs A+E on a stored dataset, writes `results/<name>.json`
- `src/compare_runs.py` – tabulates result files side by side
- `src/run_mbhb1_windows.py` – runs the network over the Spritz `mbhb1` validation event
- `results/` – outputs of runs (`compare_results*.json` are the earlier pilot runs, older format)
- `data/` – **not tracked**; inputs and generated datasets go here

## Data
Download the LDC2 Spritz training files (`LDC2_spritz_mbhb1_training_v2.h5`, etc.) from the
[LISA Data Challenge](https://zenodo.org/records/7436568) and place them in `data/`. `dataset.py` needs `mbhb1` to measure the noise PSD.

## Usage (run everything from the repo root)
```
pip install -r requirements.txt

# 1. Generate datasets ONCE. 1 event = 20 windows (10 with the merger + 10 noise only),
#    so 250k events = 2.5M merger windows + 2.5M noise windows = 5M total
#    (~330 GB as float16 -- check your disk). This follows the paper's 2.5M + 2.5M.
python src/dataset.py --events 250000 --seed 0 --out data/train_250k
python src/dataset.py --events 5000   --seed 1 --out data/test_5k      # different seed!

# 2. Train/evaluate as many hyperparameter setups as you like on that same data
python src/train_compare.py --train data/train_250k --test data/test_5k --name baseline
python src/train_compare.py --train data/train_250k --test data/test_5k --name lr3e-4 --lr 3e-4
python src/train_compare.py --train data/train_250k --test data/test_5k --name wide \
        --filters 64,128,128 --dropout 0.2 --epochs 12 --mixed-precision

# 3. Compare
python src/compare_runs.py results/baseline.json results/lr3e-4.json results/wide.json
```
Hyperparameter flags: `--epochs --batch-size --lr --filters --convs-per-block --dense --dropout`.
Other useful ones: `--max-train-shards N` (train on a subset, e.g. to see how results scale with
data), `--mixed-precision` (float16 on RTX tensor cores), `--require-gpu`, `--save-model`.

**Datasets.** `dataset.py` writes `data/<name>/manifest.json` (all generation settings and the
seed) plus `shard_*.npy` files of 250 events each. It is resumable (re-run the same command and
finished shards are skipped) and refuses to continue into a directory generated with different
settings. To use several processes, run `--part 0/4`, `--part 1/4`, ... (one per terminal).
Every results file records the manifest of the train and test sets it used.
Validation is the last 10% of the *training shards* (whole events), not random windows, so an
event's windows never appear in both training and validation.

**GPU.** `dataset.py --device auto` (default) uses CUDA if CuPy and a CUDA device are present.
Noise generation and spectrograms run on the GPU. The `bbhx` waveform also runs on the GPU if
the CUDA build of `bbhx` is installed; otherwise add `--bbhx-cpu` (the waveform stays on the CPU,
everything else is still on the GPU). Training uses a GPU automatically if TensorFlow can see
one: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
