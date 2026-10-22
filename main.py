# main.py
import sys
import os
os.environ["KERAS_BACKEND"] = "torch"
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QRadioButton, QGroupBox,
    QButtonGroup, QFrame, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from inference_worker import CameraWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐾 Clasificador de Perros y Gatos")
        self.resize(1200, 800)

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.prediction_ready.connect(self.on_prediction)
        self.worker.error.connect(self.on_error)
        self.worker.model_loaded.connect(self.on_model_loaded)

        self._build_ui()
        self.worker.finished.connect(self.on_worker_finished)
        self._cargar_modelo_inicial()

    # ── Construcción de UI ──────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ── Panel izquierdo: video ──────────────────────────────────
        left = QVBoxLayout()
        self.video_label = QLabel("Cámara inactiva")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color:#111; color:#aaa; font-size:16px;")
        left.addWidget(self.video_label)

        # Estado inferior
        self.status_label = QLabel("Modelo: — | FPS: —")
        self.status_label.setStyleSheet("color:#666; padding:4px;")
        left.addWidget(self.status_label)

        main_layout.addLayout(left, stretch=2)

        # ── Panel derecho: controles ───────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        # Grupo: selector de modelo
        grupo_modelo = QGroupBox("Modelo")
        grupo_modelo.setFont(QFont("Arial", 11, QFont.Bold))
        vbox_modelo = QVBoxLayout()

        self.btn_group = QButtonGroup(self)

        self.radio_transfer = QRadioButton(
            "Transfer Learning (MobileNetV2)\n98.61% accuracy")
        self.radio_transfer.setChecked(False)
        self.radio_transfer.setFont(QFont("Arial", 10))
        self.radio_transfer.toggled.connect(
            lambda checked: checked and self.cambiar_modelo("transfer"))

        self.radio_vit = QRadioButton(
            "Vision Transformer (desde cero)\n72.93% accuracy")
        self.radio_vit.setFont(QFont("Arial", 10))
        self.radio_vit.toggled.connect(
            lambda checked: checked and self.cambiar_modelo("vit"))

        self.radio_cnn = QRadioButton(
            "CNN end-to-end\n91.01% accuracy")
        self.radio_cnn.setFont(QFont("Arial", 10))
        self.radio_cnn.setChecked(True)
        self.radio_cnn.toggled.connect(
            lambda checked: checked and self.cambiar_modelo("cnn"))

        self.btn_group.addButton(self.radio_transfer)
        self.btn_group.addButton(self.radio_vit)
        self.btn_group.addButton(self.radio_cnn)

        for rb in (self.radio_transfer, self.radio_vit, self.radio_cnn):
            vbox_modelo.addWidget(rb)
        grupo_modelo.setLayout(vbox_modelo)
        right.addWidget(grupo_modelo)

        # Línea separadora
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("color:#ccc;")
        right.addWidget(line1)

        # Grupo: predicción
        grupo_pred = QGroupBox("Predicción")
        grupo_pred.setFont(QFont("Arial", 11, QFont.Bold))
        vbox_pred = QVBoxLayout()

        self.pred_label = QLabel("—")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.pred_label.setStyleSheet("color:#888; padding:16px;")
        vbox_pred.addWidget(self.pred_label)

        self.conf_bar = QFrame()
        self.conf_bar.setFixedHeight(18)
        self.conf_bar.setStyleSheet(
            "background-color:#eee; border-radius:4px;")
        self.conf_bar_inner = QFrame(self.conf_bar)
        self.conf_bar_inner.setFixedHeight(18)
        self.conf_bar_inner.setStyleSheet(
            "background-color:#3b82f6; border-radius:4px;")
        self.conf_bar_inner.setFixedWidth(0)
        vbox_pred.addWidget(self.conf_bar)

        grupo_pred.setLayout(vbox_pred)
        right.addWidget(grupo_pred)

        # Línea separadora
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color:#ccc;")
        right.addWidget(line2)

        # Grupo: detalles
        grupo_det = QGroupBox("Detalles")
        grupo_det.setFont(QFont("Arial", 11, QFont.Bold))
        vbox_det = QVBoxLayout()

        self.det_model = QLabel("Modelo: —")
        self.det_inf = QLabel("Inferencia: —")
        self.det_fps = QLabel("FPS: —")
        for lbl in (self.det_model, self.det_inf, self.det_fps):
            lbl.setFont(QFont("Arial", 10))
            vbox_det.addWidget(lbl)

        grupo_det.setLayout(vbox_det)
        right.addWidget(grupo_det)

        # Botones de control
        self.btn_start = QPushButton("▶ Iniciar cámara")
        self.btn_start.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_start.setStyleSheet(
            "background-color:#22c55e; color:white; padding:8px; border-radius:6px;")
        self.btn_start.clicked.connect(self.iniciar_camara)

        self.btn_stop = QPushButton("⏸ Detener")
        self.btn_stop.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_stop.setStyleSheet(
            "background-color:#ef4444; color:white; padding:8px; border-radius:6px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.detener_camara)

        self.btn_capture = QPushButton("📸 Capturar frame")
        self.btn_capture.setFont(QFont("Arial", 11))
        self.btn_capture.setStyleSheet(
            "background-color:#64748b; color:white; padding:8px; border-radius:6px;")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.capturar_frame)

        right.addWidget(self.btn_start)
        right.addWidget(self.btn_stop)
        right.addWidget(self.btn_capture)
        right.addStretch()

        main_layout.addLayout(right, stretch=1)

        # Estado de errores
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:#ef4444; font-size:11px;")
        right.addWidget(self.error_label)

    # ── Carga inicial ───────────────────────────────────────────────
    def _cargar_modelo_inicial(self):
        self.status_label.setText("Cargando modelo inicial...")
        if not self.worker.load_model("cnn"):
            self.status_label.setText("No se pudo cargar CNN; revisa el error o selecciona otro modelo.")
        self.btn_start.setEnabled(self.worker._model is not None)

    # ── Slots ───────────────────────────────────────────────────────
    @pyqtSlot()
    def iniciar_camara(self):
        if self.worker._model is None:
            self.on_error("Primero carga un modelo correctamente.")
            return
        if not self.worker.isRunning():
            self.error_label.clear()
            self.worker._running = True
            self.worker.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_capture.setEnabled(True)
            self.status_label.setText("Cámara activa")

    @pyqtSlot()
    def detener_camara(self):
        if not self.worker.stop():
            self.on_error("La cámara sigue procesando. Espera y vuelve a detenerla.")
            return
        self.btn_start.setEnabled(self.worker._model is not None)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.status_label.setText("Cámara detenida")
        self.video_label.setText("Cámara inactiva")

    @pyqtSlot()
    def capturar_frame(self):
        pixmap = self.video_label.pixmap()
        if pixmap is not None and not pixmap.isNull():
            import os, datetime
            os.makedirs("capturas", exist_ok=True)
            fname = datetime.datetime.now().strftime("capturas/frame_%Y%m%d_%H%M%S.png")
            pixmap.save(fname)
            self.status_label.setText(f"Guardado: {fname}")

    @pyqtSlot(object)
    def on_frame(self, frame):
        rgb = frame[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(str, float, float)
    def on_prediction(self, clase, conf, ms):
        color = "#22c55e" if conf > 90 else "#eab308" if conf > 70 else "#ef4444"
        icono = "🐕" if clase == "dog" else "🐈"
        self.pred_label.setText(f"{icono} {clase.upper()}")
        self.pred_label.setStyleSheet(f"color:{color}; padding:16px;")
        self.det_inf.setText(f"Inferencia: {ms:.0f} ms")
        self.det_fps.setText(f"FPS: {1000/ms:.1f}" if ms > 0 else "FPS: —")
        # Barra
        max_w = self.conf_bar.width()
        w = int(max_w * conf / 100)
        self.conf_bar_inner.setFixedWidth(w)
        self.conf_bar_inner.setStyleSheet(
            f"background-color:{color}; border-radius:4px;")

    @pyqtSlot(str)
    def on_error(self, msg):
        self.error_label.setText(msg)

    @pyqtSlot(str, str)
    def on_model_loaded(self, nombre, acc):
        self.error_label.clear()
        self.status_label.setText(f"Modelo listo: {nombre} ({acc})")
        self.det_model.setText(f"Modelo: {nombre}")

    # ── Cambio de modelo ────────────────────────────────────────────
    def cambiar_modelo(self, nombre):
        estaba_activa = self.worker.isRunning()
        self.status_label.setText(f"Cambiando a {nombre}...")
        if not self.worker.stop():
            self.on_error("La cámara sigue procesando. Espera y vuelve a intentarlo.")
            previous = {"keras": self.radio_transfer, "torch_vit": self.radio_vit,
                        "torch_cnn": self.radio_cnn}.get(self.worker._framework)
            if previous is not None:
                buttons = self.btn_group.buttons()
                for button in buttons:
                    button.blockSignals(True)
                previous.setChecked(True)
                for button in buttons:
                    button.blockSignals(False)
            return
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.error_label.clear()
        self.pred_label.setText("—")
        self.conf_bar_inner.setFixedWidth(0)
        self.video_label.setText("Cámara inactiva")
        self.det_inf.setText("Inferencia: —")
        self.det_fps.setText("FPS: —")
        loaded = self.worker.load_model(nombre)
        self.btn_start.setEnabled(loaded)
        if not loaded:
            self.status_label.setText("No se pudo cargar el modelo; revisa el error.")
            self.det_model.setText("Modelo: —")
        elif estaba_activa:
            self.iniciar_camara()

    @pyqtSlot()
    def on_worker_finished(self):
        # Una señal pendiente del hilo anterior puede llegar tras un reinicio.
        if self.worker.isRunning():
            return
        self.btn_start.setEnabled(self.worker._model is not None)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)

    def closeEvent(self, event):
        if self.worker.stop():
            event.accept()
        else:
            self.on_error("Espera a que termine la inferencia y vuelve a cerrar.")
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


# .\.venv\Scripts\python.exe App-3/verificar_modelos.py