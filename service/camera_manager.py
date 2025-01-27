import logging
import os
import queue
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from definitions import SAVE_PATH_WEBCAM
from service.detector import Detector


def generate_output_path(filename: str):
    """Генерация пути для сохранения видео с использованием текущей даты и времени."""
    os.makedirs(SAVE_PATH_WEBCAM, exist_ok=True)  # Создаём папку, если её нет
    # Форматируем название файла как день-месяц-год: часы24-минуты-секунды
    timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
    filename_1 = f'{filename}_detect_{timestamp}.mp4'
    output_path = os.path.join(SAVE_PATH_WEBCAM, filename_1)
    print('Generated output path:', output_path)
    return output_path


class CameraManager(QThread):
    img_signal = Signal(np.ndarray)
    error_signal = Signal(str)
    connect_successful_signal = Signal(str)
    log_signal = Signal(str)

    def __init__(self, detector: Detector):
        super().__init__()  # Initialize QThread
        self.frame_queue = queue.Queue()
        self.detector = detector  # Initialize Detector instance
        self.cam: cv2.VideoCapture | None = None
        self.stream = False
        self._save_video = False
        self.writer_thread = None  # Поток для записи видео

    @property
    def save_video(self):
        return self._save_video

    @save_video.setter
    def save_video(self, value):
        self._save_video = value

    def stop_stream(self):
        self.stream = False

    def connect_cam(self, num_cam: int = 0):
        self.log_signal.emit(f'Подключение к камере [{num_cam}] - connect_cam')
        time.sleep(0.5)
        cap = cv2.VideoCapture(num_cam)
        if not cap.isOpened():
            self.error_signal.emit(f'Не удалось открыть камеру index - [{num_cam}]. Проверьте камеру')
            self.cam = None
        else:
            self.connect_successful_signal.emit(f'Камера подключена: [{num_cam}] \n'
                                                f'width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} \n'
                                                f'height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
            self.cam = cap



    def disconnect_cam(self):
        if self.cam is not None:
            self.stream = False
            self.cam.release()
            self.cam = None



    def run(self):
        self.log_signal.emit(f'поток CameraManager, native id: {threading.get_native_id()} запущен')
        if self.cam is None:
            self.connect_cam(0)

        if self.cam is not None:
            self.stream = True
            # Получаем параметры исходного видео

            frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_cam = int(self.cam.get(cv2.CAP_PROP_FPS))
            real_fps = 0
            start_time = time.time()
            counter = 0

            while self.stream:
                ret, frame = self.cam.read()
                if not ret:
                    break


                if self.save_video and self.writer_thread is None:
                    # Генерируем путь для сохранения видео
                    output_path = generate_output_path('video_ss')
                    self.log_signal.emit(f'Видео сохраняется в {output_path}')
                    # Запускаем поток для записи видео
                    self.writer_thread = threading.Thread(target=self._write_video,
                                                          args=(fps_cam, frame_width, frame_height, output_path))
                    self.writer_thread.start()
                    self.log_signal.emit(f'created new thread: {self.writer_thread.native_id}')

                if ret:
                    frame = self.detector.detectYolo(frame)
                    counter += 1
                    if counter % 10 == 0:
                        real_fps = counter / (time.time() - start_time)
                        real_fps = round(real_fps, 2)
                        counter = 0
                        start_time = time.time()


                    real_fps = str(real_fps)
                    cv2.putText(frame, f"FPS: {real_fps}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
                    self.img_signal.emit(frame)
                    if self.save_video:
                        self.frame_queue.put(frame)

        # self.cam.release()
        self.log_signal.emit(f'поток CameraManager, native id: {threading.get_native_id()} завершён')
        self.disconnect_cam()

        # Завершаем поток записи
        # self.stream = False
        if self.save_video or self.writer_thread is not None:
            self.writer_thread.join()
            self.writer_thread = None


    def _write_video(self, fps, frame_width, frame_height, output_path):
        if not output_path:
            return
        # Настраиваем видеозапись
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while self.stream or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
                out.write(frame)
            except queue.Empty as e:
                print('timout, empty')

        out.release()
        self.log_signal.emit(f'Видео сохранено: {output_path}')
