# Llama 3.2 1B on the phone with ExecuTorch

What we used, and what worked on our Samsung S20 FE 5G (Snapdragon 865, arm64, Android) with Termux.

- Model: prebuilt ExecuTorch program `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` (1.14 GB) and `tokenizer.model` from Hugging Face `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET`. No export step needed.
- Runner: `llama_main` from ExecuTorch **release/1.6**, cross-compiled on an Ubuntu 22.04 laptop with the Android NDK for `arm64-v8a`, XNNPACK backend.

## 1. Phone: Termux and model files
Install Termux from F-Droid (not the Play Store), then:
```
pkg update -y && pkg upgrade -y
```
```
termux-setup-storage
```
```
pkg install -y wget
```
```
mkdir -p ~/llama && cd ~/llama
```
```
wget https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/resolve/main/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte
```
```
wget https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/resolve/main/tokenizer.model
```
Accept the Llama 3.2 licence on Hugging Face first. If the download returns 401/403, add `--header="Authorization: Bearer <your HF read token>"`.

## 2. Laptop (Ubuntu 22.04): build tools
```
sudo apt update && sudo apt install -y git ninja-build build-essential python3.11 python3.11-venv unzip wget file
```
CMake 3.31 and the Android NDK (`ANDROID_NDK` pointing to its folder) are needed.

## 3. ExecuTorch source
```
git clone https://github.com/pytorch/executorch.git && cd executorch
```
```
git checkout release/1.6 && git submodule update --init --recursive
```
```
python3.11 -m venv .venv && source .venv/bin/activate
```
```
./install_executorch.sh
```

## 4. Build ExecuTorch for Android arm64
```
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 \
  -DCMAKE_INSTALL_PREFIX=cmake-out-android -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_ENABLE_LOGGING=1 -DPYTHON_EXECUTABLE=python \
  -DBUILD_TESTING=OFF -Bcmake-out-android .
```
```
cmake --build cmake-out-android -j4 --target install --config Release
```
`-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON` was required on release/1.6.

## 5. Build the llama runner
```
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 \
  -DCMAKE_INSTALL_PREFIX=cmake-out-android -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=python -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DBUILD_TESTING=OFF \
  -Bcmake-out-android/examples/models/llama examples/models/llama
```
```
cmake --build cmake-out-android/examples/models/llama -j4 --config Release
```
```
file cmake-out-android/examples/models/llama/llama_main
```
Expect `ELF 64-bit ... ARM aarch64 ... for Android`.

## 6. Copy to the phone
Copy `llama_main` and the NDK's `libc++_shared.so` (`$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so`) to the phone's Download folder, then in Termux:
```
cp ~/storage/downloads/llama_main ~/storage/downloads/libc++_shared.so ~/llama/ && chmod +x ~/llama/llama_main
```
Binaries must live in Termux home: shared storage is not executable.

## 7. Test run (airplane mode on)
```
cd ~/llama && LD_LIBRARY_PATH=. ./llama_main --model_path Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte --tokenizer_path tokenizer.model --cpu_threads=4 --prompt "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Say hello in one line.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
```
The runner prints the answer and a stats block (prompt and generated tokens, time to first token, tokens/s).

Measured on the S20 FE: ~22 tok/s decode, ~125 tok/s prefill, ~0.5 s time to first token, 3.7 to 4.0 s model load, ~1.37 GB peak RSS.

`qc_inspector.py` calls this runner as a subprocess for each failed image with `--cpu_threads 4 --seq_len 1024 --temperature 0`.
