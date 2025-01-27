import logging
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from definitions import SAVE_PATH_FOLDER
from service.detector import Detector


def generate_output_path(filepath: str, name_folder):
    """Генерация пути для сохранения видео с использованием текущей даты и времени."""
    file_name = Path(filepath).stem
    file_extension = Path(filepath).suffix[1:]
    # Форматируем название файла как день-месяц-год: часы24-минуты-секунды
    filename = f'{file_name}_detect.{file_extension}'
    output_path = os.path.join(name_folder, filename)
    return output_path

class FoldersManager(QThread):
    img_signal = Signal(np.ndarray)
    finished_signal = Signal()
    log_signal = Signal(str)

    def __init__(self, detector: Detector):
        super(FoldersManager, self).__init__()
        self.detector = detector  # Initialize Detector instance
        self._images_path = None
        self.stream = False
        self.frame_queue = queue.Queue()
        self.writer_thread = None
        self._save_photo = False

    @property
    def save_photo(self):
        return self._save_photo

    @save_photo.setter
    def save_photo(self, value):
        self._save_photo = value

    @property
    def images_path(self):
        return self._images_path

    @images_path.setter
    def images_path(self, value):
        self._images_path = value

    def stop_stream(self):
        self.stream = False
        self.log_signal.emit('stop stream FoldersManager')

    def run(self):
        self.stream = True
        current_index =0
        counter = 0
        start_time = time.time()
        real_fps = 0

        if self.images_path:
            while self.stream:
                if current_index == len(self.images_path):
                    self.log_signal.emit('Завершено')
                    break

                if self.save_photo:
                    if self.writer_thread is None:
                        self.writer_thread = threading.Thread(target=self._save_photo_thread)
                        self.writer_thread.start()
                        self.log_signal.emit('Поток для сохранения фото запущен')

                image_path = self.images_path[current_index]

                image = cv2.imread(image_path)
                image = self.detector.detectYolo(image)

                counter += 1
                if counter % 10 == 0:
                    real_fps = counter / (time.time() - start_time)
                    real_fps = round(real_fps, 2)
                    counter = 0
                    start_time = time.time()

                real_fps = str(real_fps)
                cv2.putText(image, f"FPS: {real_fps}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)

                self.img_signal.emit(image)

                if self.save_photo:
                    self.frame_queue.put((image, image_path))

                current_index += 1

            self.stream = False
            self.finished_signal.emit()
            self.writer_thread = None

    def _save_photo_thread(self):
        count_img = 0
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        new_folder = os.path.join(SAVE_PATH_FOLDER, timestamp)
        os.makedirs(new_folder, exist_ok=True)  # Создаём папку, если её нет

        while self.stream or not self.frame_queue.empty():
            try:
                frame, path_photo = self.frame_queue.get(timeout=1)
                # print(type(frame))
                if type(frame)==np.ndarray:
                    new_filename = generate_output_path(path_photo, new_folder)
                    cv2.imwrite(new_filename, frame)
                    # self.log_signal.emit(f'фото {new_filename} сохранено')
                    count_img+=1
            except queue.Empty as e:
                print('timout, empty')
        self.log_signal.emit(f'Все фото сохранены, кол-во: {count_img}, путь: {new_folder}')


