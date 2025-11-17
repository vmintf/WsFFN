try:
    import torch
except (ImportError, ModuleNotFoundError):
    raise ImportError("PyTorch is required to use WsFFN.\nPlease install it using: `pip install torch`")

__version__ = "0.0.1"
__author__ = "민성 Skystarry"

from .WsFFN import wsFFN, Config

__all__ = ["wsFFN", "Config"]
