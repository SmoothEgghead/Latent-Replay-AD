# Latent Replay Visual Anomaly Detection on Raspberry Pi 5

A convolutional autoencoder that learns to flag visual defects from "good"
images only, with **latent replay** so it can keep learning new product
categories without forgetting old ones — and runs end-to-end on a Raspberry
Pi 5 (2 GB).

![Hazelnut AUC across training rounds](results/auc_hazelnut_focus.png)

*Hazelnut detection AUC across five sequential training rounds. Without
replay the model forgets the first category; with replay it holds, on both
the laptop and the Pi.*

ENGR 859, Spring 2026.

---

## What it is

Anomaly detection on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using reconstruction error: train the autoencoder only on defect-free
images, score a test image by the max over 8×8 patches of the per-pixel MSE,
and an unusually high score means an anomaly.

The continual-learning piece is the contribution. After the first category is
trained, the encoder is frozen and ~200 latent vectors per category are kept
in a small replay buffer. When a new category arrives, the loss combines a
reconstruction term on new images with a distillation term that keeps the
decoder's output on replayed latents close to a frozen decoder snapshot:

$$
\mathcal{L} = \mathrm{MSE}(x, D(E(x))) + \lambda \cdot \mathrm{MSE}(D_{snap}(z), D(z))
$$

No raw images are stored — only latents and one decoder snapshot — so the
memory cost stays small enough for the Pi.

## Results

AUC per category after each training round on the laptop (with replay,
λ = 3). Higher is better; 0.5 is random.

| Category   | R1 Hazelnut | R2 Bottle | R3 Grid | R4 Screw | R5 Toothbrush |
|------------|:-----------:|:---------:|:-------:|:--------:|:-------------:|
| Hazelnut   | 0.97        | 0.77      | 0.94    | 0.98     | 0.42          |
| Bottle     |             | 0.57      | 0.67    | 0.31     | 0.49          |
| Grid       |             |           | 0.43    | 0.76     | 0.12          |
| Screw      |             |           |         | 0.97     | 1.00          |
| Toothbrush |             |           |         |          | 0.48          |

The full three-condition comparison (no replay vs. with replay on laptop vs.
on Pi) is in [`results/auc_comparison_3panel.png`](results/auc_comparison_3panel.png).

## Setup

```bash
git clone <this repo>
cd <repo>
pip install -r requirements.txt
```

Then download the MVTec AD dataset and unzip it into `mvtec/` at the repo
root, so the layout is `mvtec/hazelnut/`, `mvtec/bottle/`, etc.

## How to run

There are four entrypoints, each a short top-level script you can run with
plain `python`:

| Script | What it does |
|--------|--------------|
| `main.py`           | Trains the first category from scratch and seeds the replay buffer. |
| `main_continual.py` | Loads the existing model + buffer, trains a new category with latent replay, updates the buffer. |
| `main_no_replay.py` | Same as `main_continual.py` but without the replay term — the catastrophic-forgetting baseline. |
| `eval_only.py`      | Loads the current weights and reports AUC on a chosen category without training. |

Edit the `category = "..."` line near the top of each main script to pick
the MVTec category. Hyperparameters live in `config.yaml`.

A typical sequential run on the laptop:

```bash
python main.py             # category = "hazelnut"
python main_continual.py   # category = "bottle"
python main_continual.py   # category = "grid"
python eval_only.py        # check old categories
```

## Running on the Raspberry Pi 5

Tested on a Pi 5 with 2 GB RAM, 64-bit Raspberry Pi OS, CPU-only PyTorch
(aarch64). For memory:

- set `batch_size: 4` in `config.yaml`
- skip the `Plot()` call at the end of `main.py` (matplotlib loads the
  whole test set into memory)

Everything else runs unchanged.

## Project structure

```
.
├── main.py              # train first category
├── main_continual.py    # train next category with latent replay
├── main_no_replay.py    # baseline (no replay)
├── eval_only.py         # AUC on a category, no training
├── Model.py             # Autoencoder (encoder + decoder)
├── Trainer.py           # vanilla training/test loop
├── LatentTrainer.py     # latent-replay training loop + decoder snapshot
├── ReplayBuffer.py      # in-memory latent store, save/load
├── DataPreparation.py   # MVTec ImageFolder loaders
├── ConfigParser.py
├── Logger.py
├── Plot.py
├── config.yaml
├── requirements.txt
└── results/             # generated plots
```

## References

- Pellegrini et al., *Latent Replay for Real-Time Continual Learning*, IROS 2020.
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection*, CVPR 2019.
