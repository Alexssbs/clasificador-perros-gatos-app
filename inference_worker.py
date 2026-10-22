# inference_worker.py (actualizado con CNN)
import time
import os
from pathlib import Path

# Debe configurarse antes de importar Keras.
os.environ["KERAS_BACKEND"] = "torch"
import cv2
import numpy as np
import torch
torch.set_num_threads(min(4, torch.get_num_threads()))
from PIL import Image
from torchvision import transforms
from PyQt5.QtCore import QThread, pyqtSignal


APP_DIR = Path(__file__).resolve().parent


def model_path(filename):
    # No depende del directorio desde el que se ejecutó python.
    aliases = {"best_model.pt": ("cnn-best_model.pt", "best_model.pt"),
               "modelo_perros_gatos.keras": ("tl-modelo_perros_gatos.keras", "modelo_perros_gatos.keras")}
    names = aliases.get(filename, (filename,))
    folders = [APP_DIR / "models", APP_DIR, APP_DIR.parent / "models", APP_DIR.parent]
    candidates = [folder / name for folder in folders for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No se encontró {filename}. Colócalo en App, App/models, "
        "la raíz del proyecto o models."
    )


CLASES = ["cat", "dog"]
CLASES_ES = {"cat": "Gato", "dog": "Perro"}


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float, float)  # clase, confianza %, tiempo_ms
    error = pyqtSignal(str)
    model_loaded = pyqtSignal(str, str)               # nombre, accuracy de referencia

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._model_name = None
        self._model = None
        self._framework = None   # "keras" | "torch_vit" | "torch_cnn"

    # ── Carga de modelos ────────────────────────────────────────────
    def load_model(self, model_name: str):
        if not self.stop():
            self.error.emit("La cámara sigue procesando. Espera y vuelve a intentarlo.")
            return False
        self._model = None
        self._framework = None
        self._model_name = None
        try:
            if model_name == "transfer":
                path = model_path("modelo_perros_gatos.keras")
                import keras
                if keras.backend.backend() != "torch":
                    raise RuntimeError(
                        "Keras ya se inicializó con otro backend. "
                        "Cierra el proceso y ejecuta App/main.py de nuevo."
                    )
                model = keras.models.load_model(str(path), compile=False)
                model.to("cpu")
                model.eval()
                framework, name, acc = "keras", "Transfer Learning", "98.61%"
            elif model_name == "vit":
                # El módulo original debe reproducir exactamente el entrenamiento.
                # Su ausencia ya no impide iniciar CNN o Transfer Learning.
                if not (APP_DIR / "vit_model.py").is_file():
                    raise FileNotFoundError(
                        "Falta App/vit_model.py con load_vit_checkpoint. "
                        "Se necesita la definición original del ViT entrenado."
                    )
                from vit_model import load_vit_checkpoint
                model = load_vit_checkpoint(str(model_path("vit_dogs_cats.pth")), device="cpu")
                model = model.to("cpu").eval()
                framework, name, acc = "torch_vit", "Vision Transformer", "72.93%"
            elif model_name == "cnn":
                from cnn_model import load_cnn_checkpoint
                model, _ = load_cnn_checkpoint(str(model_path("best_model.pt")), device="cpu")
                framework, name, acc = "torch_cnn", "CNN end-to-end", "91.01%"
            else:
                raise ValueError(f"Modelo desconocido: {model_name}")
            self._model, self._framework, self._model_name = model, framework, name
            self.model_loaded.emit(name, acc)
            return True
        except Exception as e:
            self.error.emit(f"Error cargando {model_name}: {type(e).__name__}: {e}")
            return False

    def stop(self):
        self._running = False
        return self.wait(2000) if self.isRunning() else True

    # ── Inferencia ──────────────────────────────────────────────────
    def _predict(self, frame_bgr):
        t0 = time.perf_counter()
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if self._framework == "keras":
                # flow_from_dataframe usó el resize PIL nearest por defecto.
                img = np.asarray(Image.fromarray(frame_rgb).resize(
                    (224, 224), Image.Resampling.NEAREST))
                # Equivale al preprocess_input de MobileNetV2: RGB [0,255] -> [-1,1].
                img = img.astype(np.float32) / 127.5 - 1.0
                batch = np.expand_dims(img, axis=0)
                # Llamada directa: no requiere tf.data ni el runtime de TensorFlow.
                with torch.no_grad():
                    output = self._model(batch, training=False)
                if tuple(output.shape) != (1, 1):
                    raise ValueError(f"Se esperaba salida sigmoid (1,1), se obtuvo {tuple(output.shape)}")
                prob = float(output.detach().cpu().numpy()[0][0])
                if not np.isfinite(prob) or not 0.0 <= prob <= 1.0:
                    raise ValueError("La salida no es una probabilidad válida.")
                idx = 1 if prob > 0.5 else 0
                conf = prob * 100 if idx == 1 else (1 - prob) * 100

            elif self._framework == "torch_vit":
                pil = Image.fromarray(frame_rgb)
                tfm = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
                ])
                tensor = tfm(pil).unsqueeze(0)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                idx = int(probs.argmax().item())
                conf = float(probs[idx].item()) * 100

            elif self._framework == "torch_cnn":
                pil = Image.fromarray(frame_rgb)
                from cnn_model import cnn_preprocess
                tensor = cnn_preprocess(pil)   # [1, 3, 224, 224] en [0,1]
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                idx = int(probs.argmax().item())
                conf = float(probs[idx].item()) * 100

            else:
                return None

            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CLASES[idx], conf, elapsed_ms
        except Exception as e:
            self.error.emit(f"Error en inferencia: {e}")
            return None

    def _annotate(self, frame, clase, conf, ms):
        color = (0, 255, 0) if conf > 90 else (0, 255, 255) if conf > 70 else (0, 0, 255)
        label = f"{CLASES_ES[clase]}: {conf:.1f}%"
        cv2.putText(frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"{ms:.0f} ms", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        bar_w = int(conf * 3)
        cv2.rectangle(frame, (10, 85), (10 + bar_w, 100), color, -1)
        return frame

    # ── Bucle principal ─────────────────────────────────────────────
    def run(self):
        cap = None
        try:
            if self._model is None:
                self.error.emit("Selecciona y carga un modelo antes de iniciar la cámara.")
                return
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.error.emit("No se pudo abrir la cámara web.")
                return
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self.error.emit("No se pudo leer un frame de la cámara.")
                    break
                result = self._predict(frame)
                if result is None:
                    break  # Evita repetir el mismo error en cada frame.
                clase, conf, ms = result
                self.prediction_ready.emit(clase, conf, ms)
                frame = self._annotate(frame, clase, conf, ms)
                self.frame_ready.emit(frame.copy())
        except Exception as e:
            self.error.emit(f"Error de cámara: {e}")
        finally:
            self._running = False
            if cap is not None:
                cap.release()
