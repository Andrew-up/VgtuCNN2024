import logging
import os
import queue
import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from definitions import SAVE_PATH_VIDEO
from service.detector import Detector


def generate_output_path(filepath: str):
    """Генерация пути для сохранения видео с использованием текущей даты и времени."""
    os.makedirs(SAVE_PATH_VIDEO, exist_ok=True)  # Создаём папку, если её нет
    file_name = Path(filepath).stem
    # Форматируем название файла как день-месяц-год: часы24-минуты-секунды
    timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
    filename = f'{file_name}_detect_{timestamp}.mp4'
    output_path = os.path.join(SAVE_PATH_VIDEO, filename)
    return output_path


class VideoManager(QThread):
    img_signal = Signal(np.ndarray)
    log_signal = Signal(str)

    def __init__(self, detector: Detector):
        super(VideoManager, self).__init__()
        self.detector = detector  # Initialize Detector instance
        self._video_path = None
        self._output_path = None
        self.current_index = 0
        self.stream = False
        self.frame_queue = Queue()  # Очередь для кадров
        self.writer_thread = None  # Поток для записи видео
        self._save_video = False

    @property
    def save_video(self):
        return self._save_video

    @save_video.setter
    def save_video(self, value):
        self._save_video = value

    @property
    def video_path(self):
        return self._video_path

    @video_path.setter
    def video_path(self, value):
        self._video_path = value

    @property
    def output_path(self):
        return self._output_path

    def stop_stream(self):
        self.stream = False
        print('stop stream VideoManager')

    def run(self):
        self.stream = True
        # output_path = None
        if self.video_path:
            self.log_signal.emit(f'Запуск видео: {self.video_path}')
            cap = cv2.VideoCapture(self.video_path)
            print('cap FPS', cap.get(cv2.CAP_PROP_FPS))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Получаем параметры исходного видео
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.detector.flag_detect:
                self.log_signal.emit(f'Используется YOLO модель для распознавания объектов')

            while self.stream:
                ret, frame = cap.read()
                if not ret:
                    print('finish video')
                    break
                if self.save_video:
                    if self.writer_thread is None or not self.writer_thread.is_alive():
                        # Генерируем путь для сохранения видео
                        print('Генерируем путь для сохранения видео')
                        output_path = generate_output_path(self.video_path)
                        self.log_signal.emit(f'Видео сохраняется в {output_path}')
                        # Запускаем поток для записи видео
                        self.writer_thread = threading.Thread(target=self._write_video,
                                                              args=(fps, frame_width, frame_height, output_path))
                        self.writer_thread.start()

                    # time.sleep(5)

                start_time = time.time()  # Начало обработки текущего кадра
                image = self.detector.detectYolo(frame)
                # Излучаем сигнал с обработанным изображением
                self.img_signal.emit(image)

                if self.save_video:
                    # Добавляем обработанный кадр в очередь
                    self.frame_queue.put(image)

                # Ограничиваем частоту обработки до заданного fps
                frame_time = 1 / fps  # Время на один кадр
                elapsed_time = time.time() - start_time  # Время обработки кадра
                time_to_sleep = frame_time - elapsed_time  # Время, которое нужно подождать
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

            cap.release()
            self.log_signal.emit(f'Видео {self.video_path} завершено')

            # Завершаем поток записи
            self.stream = False
            if self.save_video or self.writer_thread is not None:
                self.log_signal.emit('Ждем завершения записи видео в файл')
                self.writer_thread.join()
                self.writer_thread = None
            print('Video processing complete.')
        else:
            print('No video path provided')

    def _write_video(self, fps, frame_width, frame_height, output_path):
        if not output_path:
            print('No output path provided')
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
        print('Video saved at:', output_path)
