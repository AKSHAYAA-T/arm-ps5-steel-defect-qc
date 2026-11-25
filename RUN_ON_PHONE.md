# Run on the phone (Termux, Samsung S20 FE)

The two runtimes are kept separate:
- **Vision:** YOLO11n (`yolo11n_neu_640.onnx`) with ONNX Runtime on the Arm CPU (Termux package `python-onnxruntime`, CPU execution provider, 4 threads, all graph optimisations on).
- **Advisory:** Llama 3.2 1B Instruct (`.pte`) with the ExecuTorch `llama_main` runner on the Arm CPU. The pipeline calls it only when an image fails.

OpenCV is not needed: images are loaded, letterboxed and drawn with Pillow and numpy. If OpenCV happens to be installed it is used instead.

## 1. Install (once)
```
pkg install -y python python-numpy python-pillow python-onnxruntime unzip
```
```
mkdir -p ~/qc
```

## 2. Files in ~/qc
- `qc_inspector.py` (this repository)
- `yolo11n_neu_640.onnx` and `labels.txt` (from this repository or from the Colab phone bundle)
- `test_images/` and `test_labels/` (validation images and YOLO ground truth, from the Colab phone bundle)

Llama files stay in `~/llama`: `llama_main`, `libc++_shared.so`, `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`, `tokenizer.model`. If a name differs, edit the `CONFIG` block at the top of `qc_inspector.py`.

## 3. Commands
One image, with Llama advice (also prints the JSON findings):
```
python qc_inspector.py inspect --model yolo11n_neu_640.onnx --image test_images/<file>.jpg
```
A whole folder:
```
python qc_inspector.py run --model yolo11n_neu_640.onnx --images test_images
```
Vision-only benchmark (latency per stage, FPS, memory, precision / recall / F1):
```
python qc_inspector.py bench --model yolo11n_neu_640.onnx --images test_images --labels test_labels
```
Benchmark with Llama on 20 images sampled across classes, saving boxed images:
```
python qc_inspector.py bench --model yolo11n_neu_640.onnx --images test_images --labels test_labels --llama --limit 20 --save-annotated out640
```
Flags: `--conf` (default 0.25), `--no-llama` (inspect/run: rules only), `--save-annotated <dir>`, `--verbose` (bench: print every operator card).

Every inspection is appended to `inspection_log.csv`: timestamp, image, verdict, max severity, defect list (JSON), pre/infer/post ms, Llama ms, CAUSE, ACTION, LOG, and which fields were filled by rules.

For an offline proof, turn on airplane mode before running.

## 4. Operator output (example layout)
```
============================================================
<image>:  FAIL   (max severity: <Low|Medium|High>)
  - <defect>   <severity>  conf <c>  area <a>%  <edge|centre>
  CAUSE : <from Llama>
  ACTION: <from Llama>
  LOG   : <from Llama>
  [vision pre .. / infer .. / post .. ms  llama .. ms]
```

## 5. If something fails
- **`unknown command line flag 'seq_len'` or `'temperature'`:** your `llama_main` build lacks that flag; delete those two lines in `run_llama()`.
- **Llama fails or times out:** the inspector falls back to a rule-based message and prints `[llama failed: ...]`; run `llama_main` by hand to see the error. Raw Llama output of the last call is in `llama_last_raw.txt`.
- **A Llama field is empty or cut off:** it is filled from the rules and marked in `filled_by_rules`.

## 6. What to tune
- `SEVERITY_RULES`: critical classes, area thresholds (5% High, 2% Medium) and edge margin (10%). See `severity_rules.md`.
- `conf_threshold` (0.25): lower means fewer misses but more false alarms.
- `SYSTEM_PROMPT`, `EXAMPLE_*`, `DEFECT_HINTS`: the fixed 3-line answer format, grounded in the JSON findings.
