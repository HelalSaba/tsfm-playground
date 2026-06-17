import torch.nn as nn


def activation_from_str(name: str) -> nn.Module:
    """
    Get activation function from string name.

    Parameters
    ----------
    name: str
        Name of the activation function. Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu'.

    Returns
    -------
    nn.Module
        Corresponding activation function module.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")
