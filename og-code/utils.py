import numpy as np
import torch


def get_device() -> str:
    """Checks to see if cuda, then mps, then cpu are available

    Returns:
        str: The device
    """
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"

    return device
