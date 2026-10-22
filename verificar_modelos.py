"""Prueba los pesos reales sin abrir la cámara: python App/verificar_modelos.py."""
import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import platform
import sys
import traceback
from importlib.metadata import version

import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtCore import QCoreApplication

from inference_worker import CameraWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagen", help="Imagen opcional para clasificar con los tres modelos")
    args = parser.parse_args()
    app = QCoreApplication.instance() or QCoreApplication([])
    print("Python:", platform.python_version(), "|", platform.system())
    for package in ("torch", "torchvision", "keras", "numpy", "Pillow", "PyQt5"):
        print(f"{package}: {version(package)}")
    if args.imagen:
        with Image.open(args.imagen) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        print("Imagen:", args.imagen)
    else:
        rgb = np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        print("Entrada sintética: comprueba ejecución; NO mide precisión.")
    frame = rgb[:, :, ::-1].copy()
    worker = CameraWorker()
    worker.error.connect(lambda message: print("ERROR:", message))
    failures = 0
    for name in ("cnn", "transfer", "vit"):
        try:
            if not worker.load_model(name):
                failures += 1
                continue
            result = worker._predict(frame)
            if result is None:
                failures += 1
                continue
            label, confidence, elapsed = result
            if label not in ("cat", "dog") or not np.isfinite(confidence) or not 0 <= confidence <= 100:
                raise ValueError(f"Salida inválida: {result}")
            count = (worker._model.count_params() if name == "transfer"
                     else sum(p.numel() for p in worker._model.parameters()))
            print(f"OK {name}: {count:,} parámetros; {label} {confidence:.2f}%; {elapsed:.1f} ms")
        except Exception:
            failures += 1
            traceback.print_exc()
    print("Resultado:", "LOS TRES MODELOS FUNCIONAN" if failures == 0 else f"{failures} modelo(s) con error")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
