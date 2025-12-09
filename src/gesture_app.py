import sys
import os
import json
import cv2
import numpy as np
import tensorflow as tf
import winreg as reg
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# -------------------------------
# Ensure src folder is in path
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from action_mappers import ActionMapper

# -------------------------------
# Add app to Windows startup
# -------------------------------
def add_to_startup():
    exe_path = os.path.abspath(sys.argv[0])
    try:
        key = reg.HKEY_CURRENT_USER
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        name = "GestureControlApp"
        registry_key = reg.OpenKey(key, reg_path, 0, reg.KEY_SET_VALUE)
        reg.SetValueEx(registry_key, name, 0, reg.REG_SZ, exe_path)
        reg.CloseKey(registry_key)
        print("Added to startup successfully")
    except Exception as e:
        print("Failed to add to startup:", e)

# Call once at app start
add_to_startup()

# -------------------------------
# Load model and classes
# -------------------------------
model_path = os.path.join(script_dir, "../models/best_model.h5")
model = tf.keras.models.load_model(model_path)

class_indices_path = os.path.join(script_dir, "../models/class_indices.json")
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
label_to_gesture = {v: k for k, v in class_indices.items()}

# -------------------------------
# Worker Thread for Webcam
# -------------------------------
class WebcamWorker(QThread):
    frame_data = pyqtSignal(np.ndarray, str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = cv2.VideoCapture(0)
        self.mapper = ActionMapper()

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Preprocess frame for model
            img = cv2.resize(frame, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict gesture
            pred = model.predict(img, verbose=0)
            class_idx = int(np.argmax(pred))
            gesture = label_to_gesture[class_idx]

            # Trigger action
            self.mapper.trigger(gesture)

            # Emit frame & gesture for GUI
            self.frame_data.emit(frame, gesture)

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()
        self.wait()

# -------------------------------
# Main GUI Window
# -------------------------------
class GestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Control App")
        icon_path = os.path.join(script_dir, "gesture_icon.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(100, 100, 800, 600)

        # Layout
        self.layout = QVBoxLayout()
        self.label = QLabel("Webcam Feed")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        self.status_label = QLabel("Detected Gesture: None")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.status_label)

        # Start / Stop buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_webcam)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        # System Tray
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(icon_path))
        tray_menu = QMenu()
        show_action = QAction("Show")
        quit_action = QAction("Quit")
        show_action.triggered.connect(self.show_window)
        quit_action.triggered.connect(self.quit_app)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Webcam worker
        self.worker = WebcamWorker()
        self.worker.frame_data.connect(self.update_frame)

        # Start automatically
        self.start_webcam()

    # Start webcam thread (auto-restart if already running)
    def start_webcam(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker = WebcamWorker()
            self.worker.frame_data.connect(self.update_frame)
        self.worker.start()

    # Stop webcam thread
    def stop_webcam(self):
        if self.worker.isRunning():
            self.worker.stop()

    # Update GUI frame and gesture label
    def update_frame(self, frame, gesture):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.AspectRatioMode.KeepAspectRatio))
        self.status_label.setText(f"Detected Gesture: {gesture}")

    # Minimize to tray instead of closing
    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Gesture Control",
            "App minimized to tray. Double-click tray icon to restore.",
            QSystemTrayIcon.MessageIcon.Information,
            2000
        )

    def show_window(self):
        self.show()

    def quit_app(self):
        self.stop_webcam()
        QApplication.quit()

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureApp()
    window.show()
    sys.exit(app.exec())
