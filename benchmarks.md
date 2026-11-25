# Benchmarks

Device: Samsung Galaxy S20 FE 5G, Snapdragon 865 (Arm CPU: 1× Cortex-A77-class prime + 3× A77-class + 4× A55-class), 8 GB RAM, Android, Termux. Everything on the CPU, airplane mode on.
Software: Termux native Python 3.14, ONNX Runtime 1.32 (CPU execution provider, 4 threads), ExecuTorch release/1.6 `llama_main` (XNNPACK, 4 threads).
Test set: 361 NEU-DET validation images (same split for both models, not used for training). Each run starts with 3 warm-up inferences.

## 1. Vision detector on the phone

| | YOLO11n 640 (main) | YOLO11n 320 |
|---|---|---|
| Pre-process ms (mean) | 16.7 | 5.0 |
| Inference ms (mean) | 95.7 | 27.0 |
| Post-process ms (mean) | 1.2 | 0.7 |
| **Vision total ms / image** | **113.7** | **32.7** |
| **FPS** | **8.8** | **30.5** |
| Peak RSS, Python + vision | 150 MB | 86 MB |
| Box precision (IoU ≥ 0.5, class match) | 0.760 | 0.767 |
| Box recall | 0.670 | 0.630 |
| Box F1 | 0.712 | 0.692 |
| Images with at least one detection | 351 / 361 | 333 / 361 |

## 2. Training-side validation (Colab, Ultralytics `val`)

| | 640 | 320 |
|---|---|---|
| Precision | 0.676 | 0.618 |
| Recall | 0.656 | 0.694 |
| F1 | 0.666 | 0.653 |
| mAP50 | 0.739 | 0.716 |
| mAP50-95 | 0.404 | 0.385 |
| Weakest classes (per-class score, Colab validation) | crazing 0.52, rolled-in_scale 0.49 | crazing 0.45, rolled-in_scale 0.51 |
| Strongest class | scratches 0.88 | |

Phone precision/recall differ from Colab because Ultralytics reports P/R at the confidence that maximises F1 per class, while the phone uses one fixed threshold (0.25) plus our own NMS.

## 3. Llama advisory (ExecuTorch)

| | Value |
|---|---|
| Model | Llama 3.2 1B Instruct, SpinQuant INT4 (`_EO8`), `.pte` 1.14 GB |
| Model load | 3.7 to 4.0 s |
| Time to first token | ~0.5 s |
| Prefill | ~125 tok/s |
| Decode | ~22 tok/s |
| Peak RSS | ~1.37 GB (1375 MB) |
| Total advisory time per FAIL (640 + Llama, 20 images sampled across classes) | 8.77 s mean, 9.1 s p95 |
| Vision in the same run | 128 ms per image, 147 MB |
| Images flagged in that run | 19 / 20 |

Each advisory call starts `llama_main` fresh, so the ~4 s model load is part of the 8.8 s.

## 4. False alarms on defect-free metal and threshold sweep (640 model)

False-alarm set: 40 photos taken with the S20 FE camera (3024×3024), close-ups of clean brushed metal appliance panels and matte grey metal, indoor room light, empty labels. Defect set: the 361 NEU validation images.

| conf | Defect set box P / R / F1 (TP / FP / FN) | Defective images missed | Clean photos falsely flagged | Clean photos passed |
|---|---|---|---|---|
| 0.15 | 0.673 / 0.723 / 0.697 (620 / 301 / 237) | 0 / 361 | 7 / 40 (17.5%) | 33 |
| **0.25 (default)** | **0.760 / 0.670 / 0.712** (574 / 181 / 283) | **10 / 361** | **4 / 40 (10%)** | 36 |
| 0.35 | 0.825 / 0.606 / 0.699 (519 / 110 / 338) | 39 / 361 | 3 / 40 (7.5%) | 37 |
| 0.50 | 0.889 / 0.524 / 0.659 (449 / 56 / 408) | 72 / 361 | 2 / 40 (5%) | 38 |

- Default 0.25: best F1, 10 of 361 defective parts missed, 10% false alarms. 0.15 = zero-miss mode (17.5% false alarms).
- Vision time on the defect set: 113 to 116 ms (8.7 to 8.8 FPS) at every threshold. On the 9 MP clean photos: 263 to 269 ms (3.7 to 3.8 FPS), because of JPEG decode and resize of the large images (network input is always 640×640).
- Caveats: small set (2.5 points per photo); household metal under room light, not hot-rolled strip, so part of the false alarms is domain shift.

## 5. Power and temperature (640 vision, continuous)

Battery current, voltage, level and temperature logged every 2 s from `/sys/class/power_supply/battery` (Termux shell loop). Airplane mode on, charger unplugged, screen on at fixed max brightness, wake lock held. 60 s idle, then ~5 min of continuous vision benchmarking (7 passes over the 361 test images, Llama off).

| Phase | Avg current | Avg power | Battery temp | Battery |
|---|---|---|---|---|
| Idle, screen on (60 s) | 174 mA | 0.67 W | 37.0 → 36.1 °C | 59% |
| Continuous 640 vision (308 s) | 1,425 mA | 5.30 W | 36.1 → 42.4 °C | 59% → 55% |
| Due to inference | +1,251 mA | +4.63 W | +6.3 °C | 4% in 5 min |

| Pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Vision ms / image | 113.3 | 112.8 | 113.9 | 116.5 | 117.6 | 119.1 | 120.5 |
| FPS | 8.8 | 8.9 | 8.8 | 8.6 | 8.5 | 8.4 | 8.3 |

- Energy: ~0.62 J per image total, ~0.54 J above idle (at ~116 ms/image).
- Estimate: ~3 h of non-stop inspection on the 4,500 mAh battery at 1,425 mA (screen on).
- ~6% slowdown over 5 minutes as the battery reached 42.4 °C: consistent with mild thermal throttling.
- Mitigations: trigger-based (duty-cycled) inspection, the 320 model (~3.5× less compute), INT8 / GPU / NPU (future work).
- Measured with the phone's battery gauge, not an external meter; vision only, not Llama.
