# Future Massive Black Hole Mergers

A CNN that detects inspiralling massive black hole binaries (MBHBs) in LISA TDI A/E
spectrograms, following Houba et al., [arXiv:2405.11340](https://arxiv.org/abs/2405.11340).
It extends the paper's single-channel (A) network to take A and E stacked as channels.

## Layout
- `src/detection_cnn.py` – the CNN (Keras)
- `src/simulate.py` – IMRPhenomD + LISA response waveforms via `bbhx`
- `src/spectrograms.py` – time series -> fixed-size log-power spectrogram images
- `src/dataset.py` – simulated training/test set generator
- `src/train_compare.py` – trains A-only vs A+E and compares them
- `src/run_mbhb1_windows.py` – runs the network over the Spritz `mbhb1` validation event
- `results/` – outputs of the pilot comparison
- `data/` – **not tracked**; inputs and generated sets go here

## Data
Download the LDC2 Spritz training files (`LDC2_spritz_mbhb1_training_v2.h5`, etc.) from the
LISA Data Challenge and place them in `data/`. `dataset.py` needs `mbhb1` to measure the noise PSD.

## Usage (run from the repo root)
```
pip install -r requirements.txt
python src/dataset.py 600 0 data/train_pilot.npz    # events, seed, output
python src/dataset.py 150 1 data/test_pilot.npz
python src/train_compare.py 8                        # epochs
```
Training uses a GPU automatically if TensorFlow can see one (`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`).
