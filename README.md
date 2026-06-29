# PowerThru

**A multimodal wearable system for early detection of muscle fatigue and cramp risk in athletes.**

PowerThru combines wearable biosignals (EMG, skin temperature, sweat/EDA, and
3-axis acceleration) with deep-learning models and a local LLM "coach" to
estimate an athlete's real-time cramp/fatigue risk and give actionable
recommendations.

> **Project status: early-stage research / prototype.** The repository contains
> the data-collection scripts, the model-training pipelines, and a runtime
> demo. The iOS app (`PowerThruApp/`) is currently an asset/test skeleton only —
> the application source is not yet committed.

---

## Repository layout

```
PowerThru/
├── TEST2.py                     # Live demo loop: sensor input → risk score → LLM coaching
├── collect_data.py             # Generates / records labelled sensor rows into cramp_dataset.csv
├── train_model.py              # Trains the TensorFlow BiLSTM (fatigue_bilstm.h5)
├── cramp_dataset.csv           # Small sample labelled dataset
│
├── emg_fatigue_cnn.py          # PyTorch CNN-LSTM for raw EMG (see "Known limitations")
├── all_v2_loso_normalized.py   # PyTorch BiLSTM on the WESAD dataset (LOSO cross-validation)
│
├── PowerThruApp/               # iOS app (Xcode) — assets + test stubs only for now
│   └── PowerThru/Assets.xcassets
│
├── Eclipta2.png                # Project poster / logo image
├── requirements.txt
└── README.md
```

## Architecture & data flow

There are two distinct stacks in this repo. They do **not** depend on each other.

### 1. Cramp-risk demo (TensorFlow + Ollama)

```
sensor reading            ┌──────────────────────────────────────────────┐
(sim / manual)  ──────▶   │ enrich_features()  → [emg,temp,sweat,ax,ay,az,│
                          │                       accel_mag]              │
                          └──────────────────────────────────────────────┘
                                              │
                  sliding window of 10 readings (sequence_buffer)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
            model loaded?                                         not enough data /
            (fatigue_bilstm.h5)                                   no model
                    │                                                   │
            BiLSTM .predict() → 0..100                       fallback_score() heuristic
                    └─────────────────────────┬─────────────────────────┘
                                              │
                          risk score (0–100) + trend + velocity
                                              │
                                  recommendation(score)
                                              │
                          Ollama `llama3` generates a coaching message
                                              │
                              printed to console; state appended to
                                     sensor_memory.json (last 100)
```

- **`collect_data.py`** writes labelled rows to `cramp_dataset.csv`.
- **`train_model.py`** reads that CSV, builds 10-step sequences, trains a
  bidirectional LSTM, and saves `fatigue_bilstm.h5`.
- **`TEST2.py`** loads that model (if present) for live prediction, and falls
  back to a transparent heuristic (`fallback_score`) when the model is missing
  or the sequence buffer is not yet full.

### 2. WESAD stress model (PyTorch)

`all_v2_loso_normalized.py` is a self-contained research script that loads the
public [WESAD](https://archive.ics.uci.edu/dataset/465/wesad) dataset, applies
per-subject robust normalization, builds 60-second windows, and runs
Leave-One-Subject-Out (LOSO) cross-validation on a BiLSTM-with-attention
classifier.

---

## Getting started

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.com) running locally with the `llama3`
  model pulled, for the coaching text in `TEST2.py`.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the cramp-risk demo

```bash
python TEST2.py
# Press Enter to simulate a sensor reading, then choose "sim" or "manual".
# Type "exit" to quit.
```

The demo runs without a trained model (it uses `fallback_score`) and without
Ollama (it simply skips the coaching text on connection error).

### Generate sample data and train the model

```bash
python collect_data.py     # appends a labelled row to cramp_dataset.csv
python train_model.py      # trains and writes fatigue_bilstm.h5
```

### Run the WESAD experiment

```bash
# Point the script at your local WESAD download (defaults to the original
# author's path if unset):
export WESAD_DATA_ROOT=/path/to/WESAD
python all_v2_loso_normalized.py
```

---

## Datasets

| File / source        | Used by                       | Notes                                            |
| -------------------- | ----------------------------- | ------------------------------------------------ |
| `cramp_dataset.csv`  | `train_model.py`, `TEST2.py`  | Small bundled sample. Columns: `timestamp, emg, temp, sweat, accel_x, accel_y, accel_z, accel_mag, score, label`. |
| WESAD (external)     | `all_v2_loso_normalized.py`   | Download separately; set `WESAD_DATA_ROOT`.      |

## Known limitations & roadmap

These are tracked issues in the current prototype:

- **`emg_fatigue_cnn.py` is not runnable as-is.** It imports a
  `svm_freq_baseline` module (providing `EMGDataLoader` / `EMGPreprocessor`)
  and references `FrequencyFeatureExtractor` / `PseudoLabelGenerator` that are
  not present in this repository. It is kept for reference until that module is
  added.
- **Label leakage in `train_model.py`.** The `label` column is currently
  included in the model's input features. This should be removed so the model
  learns from sensors only (left as-is to avoid silently changing model
  behaviour — see the project notes).
- **Hardcoded dataset paths.** `all_v2_loso_normalized.py` now reads
  `WESAD_DATA_ROOT` from the environment, but still falls back to a personal
  absolute path.
- **BLE ingestion is not implemented.** `TEST2.py` currently simulates or
  prompts for sensor values; the BLE pipeline is planned.
- **The iOS app source is not committed** — only assets and Xcode test stubs
  exist under `PowerThruApp/`.

---

## License

No license file is currently present. Until one is added, this code is
"all rights reserved" by default — add a `LICENSE` file to clarify usage terms.
