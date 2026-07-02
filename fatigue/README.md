# PowerThru — sEMG Muscle-Fatigue Detection Pipeline

A subject-generalizing muscle-fatigue detector built on surface EMG. It is
designed to (a) run — after int8 quantization — on an ARM Cortex-M wearable,
and (b) become one branch of PowerThru's future multimodal cramp-prediction
system.

> **Dataset status.** This package targets *"A Dataset of sEMG and
> Self-Perceived Fatigue Levels for Muscle Fatigue Analysis"* (Delsys Trigno,
> 1259 Hz, 8 muscles, 13 participants, 12 movements, labels 0/1/2). **The
> dataset is not vendored here.** The pipeline is fully runnable on a synthetic
> generator so every stage is testable without the download; drop the real
> data in and implement `load_dataset()` to produce real numbers.

## Quickstart

```bash
cd fatigue
pip install -r requirements.txt

# Run the whole pipeline on synthetic data (no dataset needed):
python scripts/inspect_dataset.py --synthetic
python scripts/train_classical.py --synthetic --model random_forest
python scripts/train_cnn.py --synthetic

# Verify the science (feature math, filters, no-leakage splits):
pytest -q
```

## Layout

```
fatigue/
├── config/default.yaml         # all tunables (mirrors fatigue/config.py)
├── fatigue/                     # the package
│   ├── config.py               # typed config + reasoning for every default
│   ├── preprocessing.py        # band-pass / notch / rectify / windowing / artifacts
│   ├── features/
│   │   ├── time_domain.py      # RMS, MAV, IEMG, SSI, VAR, WL, ZC, SSC, WAMP, Hjorth
│   │   └── freq_domain.py      # MNF, MDF, peak freq, spectral entropy, band power
│   ├── data.py                 # loading interface, windowing, LOSO/GroupKFold, scaler
│   ├── models/
│   │   ├── classical.py        # RF / XGBoost / SVM / logreg / grad-boost
│   │   └── cnn.py              # compact separable 1D-CNN (+ embedding for fusion)
│   ├── train.py                # CNN training loop (Step 8 checklist)
│   ├── evaluate.py             # full metric suite + per-subject/movement
│   └── utils.py                # seeding, logging
├── scripts/                     # inspect / train_classical / train_cnn
└── tests/                       # feature & preprocessing correctness (pytest)
```

## Design decisions (and why)

**Preprocessing.** 20–450 Hz zero-phase Butterworth band-pass (removes DC and
motion artifact below 20 Hz; upper edge below the 629.5 Hz Nyquist and above
most myoelectric energy) + a 60 Hz power-line notch (set 50 Hz for EU data).
Zero-phase `filtfilt` avoids the group delay that would misalign the signal
with time-synced labels. Spectral features are computed on the band-passed,
**non-rectified** signal — rectification distorts the spectrum and would
corrupt median/mean-frequency, the primary fatigue markers.

**Windowing.** 0.5 s windows (~630 samples) at 50% overlap: long enough for a
low-variance Welch PSD, short enough to track fatigue during dynamic movement.

**Features.** The physiological fatigue signature is *spectral compression*
(MDF/MNF fall) plus rising amplitude. Frequency-domain features are therefore
the headline inputs; time-domain amplitude/complexity features are
complementary. All feature code is NumPy-only and cheap, so it ports to the MCU.

**Labels (Step 4).** Default is **binary** (`level >= 1` = fatigued). Rationale:
self-perceived 0/1/2 labels are subjective and the moderate/high boundary is
noisy across subjects; binary detection is the more robust, higher-value signal
for a safety-oriented wearable. `scheme: three_class` is available for the
finer-grained task, and an ordinal-regression head is a natural future option.

**Splitting (Step 5).** Leave-One-Subject-Out. Each fold's test subject is
entirely unseen — exactly the deployment condition (a new athlete). Random
window splits would leak subject identity and massively inflate scores. The
feature scaler and per-channel normalization are fit on **training subjects
only**. `GroupKFold` is provided for larger cohorts.

**Models (Steps 6–7).** Classical baselines on engineered features
(Random Forest / XGBoost / SVM / logistic regression) and a compact
depthwise-separable **1D-CNN** on raw windows. *Recommended production model:*
start with the Random Forest / gradient-boosting baseline — with only ~13
subjects it is more sample-efficient, interpretable, and trivially deployable;
graduate to the quantized CNN if it wins on LOSO. The CNN exposes a fixed-width
embedding (`FatigueCNN.embed`) precisely so it can feed the future fusion model.

**Training (Step 8).** Seeded RNGs, class-balanced cross-entropy, AdamW weight
decay, dropout, `ReduceLROnPlateau`, gradient clipping, optional mixed
precision, and early stopping on validation balanced accuracy.

**Evaluation (Step 9).** Balanced accuracy, precision/recall/specificity, F1,
MCC, Cohen's kappa, ROC/PR-AUC, confusion matrix, and per-subject/per-movement
breakdowns — because the task is imbalanced and accuracy alone is misleading.

## Status: implemented vs. pending

| Step | Status |
|------|--------|
| 1 Dataset inspection | Script ready; run against real data once `load_dataset` is implemented. |
| 2 Preprocessing | ✅ Implemented + tested. |
| 3 Feature engineering | ✅ Time + frequency domain implemented + tested. Wavelet/nonlinear are stubbed flags (see roadmap). |
| 4 Label engineering | ✅ Binary/three-class selectable; recommendation documented. |
| 5 Split strategy | ✅ LOSO + GroupKFold, leakage-safe scaler. |
| 6 Models | ✅ Classical baselines + 1D-CNN. TCN/Transformer are roadmap. |
| 7 Production model | ✅ Recommendation documented (RF/GBM first, quantized CNN if it wins). |
| 8 Training | ✅ Full checklist in `train.py`. |
| 9 Evaluation | ✅ Full metric suite. |
| 10 Interpretability | ⏳ Needs a trained model on real data (SHAP for classical; Grad-CAM/attention for CNN). |
| 11 Deployment | ⏳ Needs a trained model to measure. Path: PyTorch → ONNX → TFLite-Micro, int8 post-training quantization; the separable-conv CNN is sized for tens of thousands of params to fit Cortex-M flash/RAM. |
| 12 Multimodal fusion | ✅ Output contract (`FatigueOutput`: probability / stage / confidence / embedding) defined for a downstream BiLSTM cramp model. |

## Known dataset risks & mitigations

- **Small cohort (13 subjects), subjective labels, high between-subject
  variability.** Mitigate with: LOSO (already default), per-subject
  normalization, class balancing, data augmentation (jitter, magnitude/time
  warping, channel dropout), and — if scores are unstable — self-supervised or
  contrastive pretraining on the raw sEMG before fine-tuning on labels.
- **Class imbalance** toward the non-fatigued state — handled via balanced loss
  and balanced-accuracy/MCC as headline metrics.

## Note on the synthetic generator

`make_synthetic_dataset` yields cleanly separable data (fatigue = deterministic
spectral shift), so it will report ~100% accuracy. That validates the plumbing,
**not** model quality. Treat any metric from synthetic data as meaningless for
research conclusions — real sEMG will be substantially harder.
