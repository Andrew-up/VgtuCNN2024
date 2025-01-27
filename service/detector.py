import logging
import os
import threading
import time

import cv2
from PySide6.QtCore import Signal, QObject


class Detector(QObject):
    finish_load_model = Signal(bool)
    log_signal = Signal(str)

    def __init__(self, model_path):
        super().__init__()
        self.th_load_model = None
        self.model_path = model_path
        self.model = None
        self.flag_detect = False
        self.count_upd = 0

    def load_model(self):
        self.finish_load_model.emit(False)
        if not os.path.exists(self.model_path):
            self.log_signal.emit(f'Ошибка. Файл модели не найден: {self.model_path}')
            return

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.finish_load_model.emit(True)

    def load_model_in_thread(self, status):
        if status:
            self.th_load_model = threading.Thread(target=self.load_model)
            self.th_load_model.start()
            self.flag_detect = True
        else:
            self.flag_detect = False


    def detectYolo(self, image):
        if not self.flag_detect or not self.model:
            return image


        if self.model:
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

            return image
