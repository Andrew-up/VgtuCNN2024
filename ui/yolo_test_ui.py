import sys
import os
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QFileDialog, QPushButton
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QTimer
from ultralytics import YOLO
import cv2
import numpy as np

def load_images_from_folder(folder_path):
    """Загружает пути ко всем изображениям в указанной папке."""
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    return [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(supported_formats)]

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with YOLO Detection")
        self.setGeometry(100, 100, 800, 600)

        # Загрузка модели YOLO
        self.model = YOLO(r"D:\myProgramm\VgtuCNN2024\model_temp_train\best2_fine_tuning_640_3_image\weights\best_ncnn_model_3img")  # Замените на вашу модель

        # Центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Основной layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Кнопка для выбора папки
        self.select_folder_button = QPushButton("Выбрать папку с изображениями")
        self.select_folder_button.clicked.connect(self.load_images)
        self.layout.addWidget(self.select_folder_button)

        # Метка для FPS
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.layout.addWidget(self.fps_label)

        # Метка для отображения изображения
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Таймер для обновления изображений
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)

        # Список изображений
        self.image_paths = []
        self.current_index = 0
        self.start_time = None
        self.frame_count = 0

    def load_images(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder_path:
            return

        self.image_paths = load_images_from_folder(folder_path)

        if not self.image_paths:
            self.image_label.setText("В выбранной папке нет изображений.")
            self.timer.stop()
            return

        # Сброс индекса и запуска таймера
        self.current_index = 0
        self.start_time = time.time()
        self.frame_count = 0
        self.timer.start()  # Устанавливаем FPS в 30 кадров в секунду

    def update_image(self):
        if not self.image_paths:
            return

        # Загружаем текущее изображение
        image_path = self.image_paths[self.current_index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Выполняем распознавание объектов с помощью YOLO
        results = self.model(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()  # Получаем результаты

        # Рисуем результаты на изображении
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Рисуем прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Добавляем метку и вероятность
            label = f"{self.model.names[int(class_id)]} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Конвертируем изображение в формат QImage
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Отображаем изображение
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Обновляем индекс изображения
        self.current_index = (self.current_index + 1) % len(self.image_paths)

        # Обновляем FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.fps_label.setText(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())
