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

### 3. Install dependencies
```bash
# Core + PyTorch (Windows)
pip install -r requirements-windows.txt

# PaddlePaddle backend for PaddleOCR (Windows CPU-only)
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Dev tools (optional)
pip install -r requirements-dev.txt
```

### 4. Verify installation
```bash
python -c "import torch; import ultralytics; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__)"
```