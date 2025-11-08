from .WsFFN import wsFFN, Config


try:
    import torch
except (ImportError, ModuleNotFoundError):
    raise ImportError("PyTorch is required to use wsFFN.\nTrying to install PyTorch: `pip install torch`")

__all__ = ["wsFFN", "Config"]
