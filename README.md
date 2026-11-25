# Models

The ONNX files and `labels.txt` are in the **repository root** (next to the main README).

| File | What | Size |
|---|---|---|
| `yolo11n_neu_640.onnx` | Main detector. YOLO11n fine-tuned on NEU-DET, 640×640 input, ONNX opset 12, static shape, no NMS inside (done in Python). Output `(1, 10, N)`: 4 box values + 6 class scores. | ~10 MB |
| `yolo11n_neu.onnx` | Same at 320×320, speed comparison. | ~10 MB |
| `labels.txt` | Class names in index order. | tiny |

Both are produced by the notebooks in `notebooks/`.

**Llama (not stored here, 1.1 GB):** download `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` and `tokenizer.model` from Hugging Face `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` (accept the Llama 3.2 licence first). See `docs/llama_executorch_setup.md`.
