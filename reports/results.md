# Results (UCI HAR (raw inertial signals))

Test set: 2947 windows from subjects never seen in training. CNN numbers are mean ± std over seeds [0, 1, 2] (40 epochs).

| Model | Accuracy % | Macro-F1 % | Weights (bytes) | CPU latency, 1 window (ms) |
|---|---|---|---|---|
| DSP features (179) + logistic regression | 94.54 | 94.46 | 5,752 | - |
| 1D CNN, FP32 (PyTorch) | 93.28 ± 0.58 | 93.30 ± 0.60 | 106,010 | 0.695 |
| 1D CNN, int8 PTQ (PyTorch x86 backend) | 93.33 ± 0.62 | 93.35 ± 0.62 | 35,400 | 1.096 |
| 1D CNN, int8 integer-only engine (NumPy, p99.99 calibration) | 93.29 ± 0.58 | 93.31 ± 0.59 | 25,016 | - |
| 1D CNN, int8 integer-only engine (NumPy, max calibration) | 93.27 ± 0.43 | 93.29 ± 0.44 | 25,016 | - |
| 1D CNN, 50% magnitude pruning (no fine-tuning) | 92.96 ± 0.39 | 92.94 ± 0.43 | - | - |
| 1D CNN, 75% magnitude pruning (no fine-tuning) | 86.25 ± 1.15 | 86.30 ± 0.96 | - | - |
| 1D CNN, 90% magnitude pruning (no fine-tuning) | 34.83 ± 6.75 | 23.05 ± 6.47 | - | - |

Agreement of int8 predictions with FP32 predictions: PyTorch int8 99.15 ± 0.20 %, NumPy engine 99.82 ± 0.06 %.

## Calibration set size (NumPy int8 engine, macro-F1 %)

| Calibration windows | Macro-F1 % |
|---|---|
| 16 | 93.36 ± 0.55 |
| 64 | 93.29 ± 0.55 |
| 256 | 93.30 ± 0.53 |
| 1024 | 93.30 ± 0.53 |

## Per-layer SQNR, seed 0 (dB, higher is better)

| conv1 | conv2 | conv3 | global avg pool | fc (logits) |
|---|---|---|---|---|
| 34.1 | 34.7 | 36.3 | 40.1 | 42.9 |

## Confusion matrix, int8 engine, seed 0 (rows = truth)

| | WALK | UPST | DOWN | SITT | STAN | LAYI |
|---|---|---|---|---|---|---|
| WALKING | 496 | 0 | 0 | 0 | 0 | 0 |
| UPSTAIRS | 6 | 461 | 4 | 0 | 0 | 0 |
| DOWNSTAIRS | 4 | 22 | 394 | 0 | 0 | 0 |
| SITTING | 0 | 5 | 0 | 372 | 94 | 20 |
| STANDING | 0 | 1 | 0 | 58 | 473 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 537 |

Platform: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel / torch 2.6.0+cu124
