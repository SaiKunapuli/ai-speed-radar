# AI Speed Radar

An AI-powered system that detects vehicle speeds from video footage using YOLO object detection, ByteTrack tracking, and perspective-corrected speed estimation.

## Setup

### 1. Create a virtual environment
```bash
python -m venv .venv
```

### 2. Activate the environment

**Windows (Git Bash / PowerShell):**
```bash
source .venv/Scripts/activate
```
**Windows (CMD):**
```cmd
.venv\Scripts\activate
```
**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

Choose your platform:

#### Windows (NVIDIA GPU)
```bash
pip install -r requirements-windows.txt
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

#### Windows (AMD GPU)
```bash
pip install -r requirements-windows-amd.txt
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```
> **Note:** AMD GPU acceleration uses DirectML via ONNX Runtime. For full ROCm PyTorch support, see the instructions in `requirements-windows-amd.txt`.

#### macOS (Apple Silicon / Intel)
```bash
pip install -r requirements-macos.txt
```
> **Note:** MPS (Metal) GPU acceleration is built into PyTorch.
> **Warning:** PaddleOCR will install but **will not work** on macOS (no paddlepaddle backend).
> After installing, run: `pip uninstall paddleocr -y`
> Use easyocr instead (already included in core requirements).

#### Dev tools (all platforms, optional)
```bash
pip install -r requirements-dev.txt
```

### 4. Check GPU
```bash
python scripts/check_gpu.py
```

### 5. Verify installation
```bash
python -c "import torch; import ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__)"
```

## Requirements Files

| File | Platform | GPU Support |
|------|----------|-------------|
| `requirements.txt` | All (core) | — |
| `requirements-windows.txt` | Windows | NVIDIA CUDA / CPU |
| `requirements-windows-amd.txt` | Windows | AMD (DirectML) |
| `requirements-macos.txt` | macOS | Apple MPS (built-in) |
| `requirements-dev.txt` | All | Dev tools (pytest, black, etc.) |