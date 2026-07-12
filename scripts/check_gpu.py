#!/usr/bin/env python3
"""
GPU Detection Utility for AI Speed Radar
Detects available GPU backends across platforms and reports compatibility.
"""

import sys


def check_cuda():
    """Check for NVIDIA CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            return {
                "available": True,
                "backend": "CUDA (NVIDIA)",
                "device_count": count,
                "device_name": name,
                "cuda_version": cuda_ver,
                "pytorch_version": torch.__version__,
            }
    except ImportError:
        pass
    return {"available": False, "backend": "CUDA (NVIDIA)"}


def check_mps():
    """Check for Apple Metal Performance Shaders (macOS)."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "available": True,
                "backend": "MPS (Apple Silicon)",
                "pytorch_version": torch.__version__,
            }
    except ImportError:
        pass
    return {"available": False, "backend": "MPS (Apple Silicon)"}


def check_rocm():
    """Check for AMD ROCm GPU."""
    try:
        import torch
        # ROCm devices show up through the CUDA API in PyTorch
        if torch.cuda.is_available():
            # Check if it's actually ROCm
            if hasattr(torch.version, "hip") and torch.version.hip:
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                return {
                    "available": True,
                    "backend": "ROCm (AMD)",
                    "device_count": count,
                    "device_name": name,
                    "hip_version": torch.version.hip,
                    "pytorch_version": torch.__version__,
                }
    except ImportError:
        pass
    return {"available": False, "backend": "ROCm (AMD)"}


def check_directml():
    """Check for DirectML (Windows - AMD/Intel GPU via ONNX Runtime)."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            # Try to get device info
            try:
                devices = ort.get_available_providers()
                return {
                    "available": True,
                    "backend": "DirectML (ONNX Runtime)",
                    "providers": providers,
                }
            except Exception:
                return {
                    "available": True,
                    "backend": "DirectML (ONNX Runtime)",
                }
    except ImportError:
        pass
    return {"available": False, "backend": "DirectML (ONNX Runtime)"}


def check_ultralytics_device():
    """Check ultralytics version."""
    try:
        import ultralytics
        return {"ultralytics_version": ultralytics.__version__}
    except ImportError:
        return {"ultralytics_version": "not installed"}


def main():
    print("=" * 55)
    print("   AI SPEED RADAR - GPU Detection Utility")
    print("=" * 55)
    print(f"   Platform: {sys.platform}")
    print(f"   Python:   {sys.version.split()[0]}")
    print("-" * 55)

    results = {}

    # Run all checks
    checks = {
        "CUDA (NVIDIA)": check_cuda,
        "MPS (Apple Silicon)": check_mps,
        "ROCm (AMD)": check_rocm,
        "DirectML (ONNX)": check_directml,
    }

    any_available = False

    for name, check_fn in checks.items():
        result = check_fn()
        status = "[AVAILABLE]" if result["available"] else "[not found]"
        print(f"\n  {name}")
        print(f"    Status: {status}")

        if result["available"]:
            any_available = True
            for key, value in result.items():
                if key not in ("available", "backend"):
                    print(f"    {key}: {value}")

    # Ultralytics info
    uinfo = check_ultralytics_device()
    if uinfo.get("ultralytics_version"):
        print(f"\n  Ultralytics: {uinfo['ultralytics_version']}")

    print("\n" + "-" * 55)

    if any_available:
        print("   *** GPU acceleration is available!")
        print("   Ready for AI speed detection.")
    else:
        print("   WARNING: No GPU detected. Running in CPU mode.")
        print("   Inference will be slower but will still work.")

    print("=" * 55)

    return 0 if any_available else 1


if __name__ == "__main__":
    sys.exit(main())
