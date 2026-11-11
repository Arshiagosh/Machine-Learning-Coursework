"""Run the saved Keras digit classifiers with NumPy only (no TensorFlow).

The two models in models/ were trained with Keras 3 on Colab. TensorFlow has no
conda-forge build for Windows, so this small reader loads the layer list and weights
from the .h5 file and does the forward pass in NumPy (inference only; dropout is a
no-op at inference). Supported layers: InputLayer, Conv2D (valid/same, stride 1),
BatchNormalization, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, Dense.

    model = load_model("models/improved_digit_classifier.h5")
    probs = model.predict(x)          # x: (N, 28, 28, 1) float in [0, 1]
"""
import json
import h5py
import numpy as np


def _act(x, name):
    if name in (None, "linear"):
        return x
    if name == "relu":
        return np.maximum(x, 0)
    if name == "softmax":
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    raise NotImplementedError(name)


def _conv2d(x, k, b, padding):
    """x: (N,H,W,C), k: (kh,kw,C,F) -> (N,H',W',F), stride 1."""
    kh, kw = k.shape[:2]
    if padding == "same":
        ph, pw = kh // 2, kw // 2
        x = np.pad(x, ((0, 0), (ph, kh - 1 - ph), (pw, kw - 1 - pw), (0, 0)))
    N, H, W, C = x.shape
    Ho, Wo = H - kh + 1, W - kw + 1
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))  # N,Ho,Wo,C,kh,kw
    cols = cols.transpose(0, 1, 2, 4, 5, 3).reshape(N, Ho, Wo, kh * kw * C)
    return cols @ k.reshape(kh * kw * C, -1) + b


def _maxpool(x, p, s):
    N, H, W, C = x.shape
    Ho, Wo = (H - p) // s + 1, (W - p) // s + 1
    out = np.full((N, Ho, Wo, C), -np.inf, x.dtype)
    for i in range(p):
        for j in range(p):
            out = np.maximum(out, x[:, i:i + s * Ho:s, j:j + s * Wo:s, :])
    return out


class NumpyKerasModel:
    def __init__(self, path):
        with h5py.File(path, "r") as h:
            cfg = json.loads(h.attrs["model_config"])
            self.layers = []
            for L in cfg["config"]["layers"]:
                c, name = L["config"], L["config"]["name"]
                w = {}
                if name in h["model_weights"]:
                    def grab(n, o, w=w):
                        if isinstance(o, h5py.Dataset):
                            w[n.split("/")[-1]] = o[()]
                    h["model_weights"][name].visititems(grab)
                self.layers.append((L["class_name"], c, w))

    def predict(self, x, verbose=0):
        x = np.asarray(x, np.float32)
        for kind, c, w in self.layers:
            if kind in ("InputLayer", "Dropout"):
                continue
            if kind == "Conv2D":
                x = _act(_conv2d(x, w["kernel"], w["bias"], c["padding"]), c["activation"])
            elif kind == "BatchNormalization":
                x = (x - w["moving_mean"]) / np.sqrt(w["moving_variance"] + c["epsilon"]) * w["gamma"] + w["beta"]
            elif kind == "MaxPooling2D":
                x = _maxpool(x, c["pool_size"][0], c["strides"][0])
            elif kind == "Flatten":
                x = x.reshape(x.shape[0], -1)
            elif kind == "GlobalAveragePooling2D":
                x = x.mean(axis=(1, 2))
            elif kind == "Dense":
                x = _act(x @ w["kernel"] + w["bias"], c["activation"])
            else:
                raise NotImplementedError(kind)
        return x


def load_model(path, compile=False):
    return NumpyKerasModel(path)
