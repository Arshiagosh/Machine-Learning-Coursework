"""Accuracy of the two saved digit classifiers on the printed-digits dataset.

Reads every image under assets/assets/0..9 (the Kaggle dataset, see data/README.md),
resizes it to 28x28 grayscale like the training code, and runs both models with the
NumPy reader. The training split (80/20, seed 123 in Keras) can't be rebuilt exactly
without TensorFlow, so this is accuracy on the whole dataset, training images included.

    python evaluate_models.py [path/to/assets/assets]
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

import keras_numpy

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("assets", "assets")
X, y = [], []
for label in range(10):
    for f in sorted(glob.glob(os.path.join(root, str(label), "*"))):
        img = Image.open(f).convert("L").resize((28, 28), Image.BILINEAR)
        X.append(np.asarray(img, np.float32) / 255.0)
        y.append(label)
X, y = np.stack(X)[..., None], np.array(y)
out = {"images": int(len(y))}
for name in ("digit_classifier", "improved_digit_classifier"):
    model = keras_numpy.load_model(os.path.join("models", name + ".h5"))
    pred = np.concatenate([model.predict(X[i:i + 256]).argmax(1) for i in range(0, len(X), 256)])
    out[name] = {"accuracy_all_images": round(float((pred == y).mean()), 4),
                 "per_class": [round(float((pred[y == k] == k).mean()), 3) for k in range(10)]}
    print(name, out[name])
json.dump(out, open("model_accuracy.json", "w"), indent=2)
