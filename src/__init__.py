try:
    import torch
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "PyTorch is required to use WsFFN.\n" "Please install it using: `pip install torch`"
    )

__name__ = "WsFFN"
__version__ = "0.0.1a2"
__author__ = "민성 Skystarry"

from .WsFFN import Config, wsFFN

__all__ = ["wsFFN", "Config"]
