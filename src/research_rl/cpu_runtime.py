"""CPU-only research branch: never query or initialize a GPU backend."""
import os


def hide_accelerators():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""


def require_cpu(device="cpu"):
    if str(device) != "cpu":
        raise ValueError("This research branch permits CPU only; GPU use is prohibited")
    import torch
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("Install the CPU-only PyTorch wheel in a separate environment")


hide_accelerators()
