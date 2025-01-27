import enum
import logging
import os
import sys
import time

import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, \
    QRadioButton, QFileDialog, QTextEdit, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem

from service.camera_manager import CameraManager
from service.detector import Detector
from service.folders_manager import FoldersManager
from service.video_manager import VideoManager
from definitions import LOG_PATH

class QtLogHandler(logging.Handler):
    """
    Кастомный обработчик для отправки логов в виджет Qt.
    """

    def __init__(self, append_text_callback):
        super().__init__()
        self.append_text_callback = append_text_callback

    def emit(self, record):
        # Форматирование сообщения
        log_entry = self.format(record)
        # Отправка сообщения в виджет через callback
        self.append_text_callback(log_entry)


def load_images_from_folder(folder_path):
    """Загружает пути ко всем изображениям в указанной папке."""
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    return [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
            f.lower().endswith(supported_formats)]


class FLAG_OPERATION(enum.Enum):
    NO_SELECT = 0
    FOLDER = 1
    VIDEO = 2
    WEBCAM = 3


class VideoWidget(QWidget):
    def __init__(self):
        super(VideoWidget, self).__init__()
        self.log_thread = None
        self.logger = logging.getLogger('MyApp')
        root_layout = QVBoxLayout()
        top_menu_layout = QHBoxLayout()
        radiobutton_folders = QRadioButton('Каталог')
        radiobutton_video_file = QRadioButton('Видеофайл')
        radiobutton_webcam = QRadioButton('Веб-камера')
        self.btn_start = QPushButton('Запустить')
        self.btn_stop = QPushButton('Остановить')
        self.btn_open_folder = QPushButton('Выбрать каталог')
        self.btn_open_video = QPushButton('Выбрать видео')

        # Настройка сцены и элемента для отображения видео
        self.scene = QGraphicsScene()
        self.video_item = QGraphicsPixmapItem()
        self.widget_QGraphicsView = QGraphicsView()
        self.widget_QGraphicsView.setMinimumWidth(680)
        self.widget_QGraphicsView.setMinimumHeight(500)
        self.widget_QGraphicsView.setScene(self.scene)

        radiobutton_webcam.clicked.connect(self.click_checkbox_webcam)
        radiobutton_folders.clicked.connect(self.click_checkbox_folders)
        radiobutton_video_file.clicked.connect(self.click_checkbox_video_file)

        self.log_text_widget = QTextEdit()
        self.log_text_widget.setMaximumHeight(100)
        self.log_text_widget.setReadOnly(True)

        top_menu_layout.addWidget(radiobutton_folders)
        top_menu_layout.addWidget(radiobutton_video_file)
        top_menu_layout.addWidget(radiobutton_webcam)
        root_layout.addLayout(top_menu_layout)
        checkbox_use_cnn = QCheckBox('Включить распознавание объекта')
        checkbox_save_detect = QCheckBox('Включить сохранение результата')
        checkbox_use_cnn.clicked.connect(lambda x: self.click_checkbox_use_cnn(checkbox_use_cnn.isChecked()))
        checkbox_save_detect.clicked.connect(
            lambda x: self.click_checkbox_save_result(checkbox_save_detect.isChecked()))

        self.btn_open_folder.clicked.connect(self.open_catalog_images)
        self.btn_open_folder.setVisible(False)

        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_open_video.setVisible(False)

        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.on_click_btn_stop)
        self.btn_start.setVisible(False)

        root_layout.addWidget(self.btn_open_folder)
        root_layout.addWidget(self.btn_open_video)
        root_layout.addWidget(checkbox_use_cnn)
        root_layout.addWidget(checkbox_save_detect)
        root_layout.addWidget(self.btn_start)
        root_layout.addWidget(self.btn_stop)

        self.scene.addItem(self.video_item)

        root_layout.addWidget(self.log_text_widget)
        root_layout.addWidget(self.widget_QGraphicsView)

        model_yolo_path = r'D:\myProgramm\VgtuCNN2024\model_temp_train\best2_fine_tuning_640_3_image\weights\best.pt'
        self.detector = Detector(model_path=model_yolo_path)
        self.detector.log_signal.connect(self.logging_other_thread)
        self.detector.finish_load_model.connect(self.set_status_load_model)

        self.camera_manager = CameraManager(detector=self.detector)
        self.folders_manager = FoldersManager(detector=self.detector)
        self.video_manager = VideoManager(detector=self.detector)

        self.camera_manager.log_signal.connect(self.logging_other_thread)
        self.folders_manager.log_signal.connect(self.logging_other_thread)
        self.video_manager.log_signal.connect(self.logging_other_thread)



        self.camera_manager.img_signal.connect(self.set_image_scene)
        self.camera_manager.connect_successful_signal.connect(self.cam_connection_successful)
        self.camera_manager.error_signal.connect(self.cam_connection_error)

        self.folders_manager.img_signal.connect(self.set_image_scene)
        self.folders_manager.finished_signal.connect(self.finish_process_detect)

        self.video_manager.img_signal.connect(self.set_image_scene)

        self.flags_current_operation = FLAG_OPERATION.NO_SELECT

        self.fixed_width = 640
        self.btn_start.clicked.connect(self.on_click_btn_run)

        self.setLayout(root_layout)
        self.setWindowTitle('Yolo detect')

        # self.log_queue = queue.Queue()
        self.setup_logger()

    def setup_logger(self):
        os.makedirs(LOG_PATH, exist_ok=True)
        self.logger = logging.getLogger("MyApp")
        self.logger.setLevel(logging.DEBUG)

        # Формат логирования
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        # Обработчик для файла (явно указываем кодировку UTF-8)
        file_handler = logging.FileHandler(f"{LOG_PATH}/{timestamp}_run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Обработчик для виджета Qt
        qt_handler = QtLogHandler(self.append_text_to_log)
        qt_handler.setLevel(logging.DEBUG)
        qt_handler.setFormatter(formatter)
        self.logger.addHandler(qt_handler)

    def finish_process_detect(self):
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)

    def set_status_load_model(self, status: bool):
        if status:
            self.logger.info('Модель YOLO загружена')
        else:
            self.logger.info('Идёт загрузка YOLO модели... ')

    def logging_other_thread(self, msg):
        self.logger.info(msg)

    def append_text_to_log(self, text):
        # self.logger.info(text)
        self.log_text_widget.append(text)

    def set_image_scene(self, image):
        if type(image) == np.ndarray:
            height, width, channel = image.shape

            # Вычисление новой высоты для сохранения соотношения сторон
            scale_factor = self.fixed_width / width
            new_width = self.fixed_width
            new_height = int(height * scale_factor)
            resized_frame = cv2.resize(image, (new_width, new_height))
            bytes_per_line = channel * new_width
            qt_image = QImage(resized_frame.data, new_width, new_height, bytes_per_line, QImage.Format_BGR888)

            self.video_item.setPixmap(QPixmap.fromImage(qt_image))

    def click_checkbox_use_cnn(self, status):
        self.logger.info(f'Режим детектирования == {status}')
        self.detector.load_model_in_thread(status)

    def click_checkbox_save_result(self, status):
        self.logger.info(f'Режим сохранения результата == {status}')
        self.video_manager.save_video = status
        self.folders_manager.save_photo = status
        self.camera_manager.save_video = status

    def on_click_btn_run(self):
        self.logger.info(f'Запуск просмотра в режиме {self.flags_current_operation}')
        if self.flags_current_operation == FLAG_OPERATION.WEBCAM:
            self.camera_manager.stream = False
            self.camera_manager.start()

        if self.flags_current_operation == FLAG_OPERATION.FOLDER:
            self.folders_manager.stream = False
            self.folders_manager.start()

        if self.flags_current_operation == FLAG_OPERATION.VIDEO:
            self.video_manager.stream = False
            self.video_manager.start()

        if self.flags_current_operation != FLAG_OPERATION.NO_SELECT:
            self.btn_stop.setVisible(True)
            self.btn_start.setVisible(False)

    def on_click_btn_stop(self):
        self.folders_manager.stop_stream()
        self.video_manager.stop_stream()
        self.camera_manager.stop_stream()
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)
        self.logger.info(f'Остановка: {self.flags_current_operation}')

    def cam_connection_successful(self, message):
        self.logger.info(message)
        self.btn_start.setVisible(False)

    def cam_connection_error(self, message):
        self.logger.info(message)
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)

    def open_video(self):
        self.flags_current_operation = FLAG_OPERATION.NO_SELECT
        video_path = QFileDialog.getOpenFileName(self, "Выберите видео с изображениями", "",
                                                 "Видео файлы (*.mp4 *.avi *.mov *.mkv);;Все файлы (*.*)")
        if video_path[0]:
            self.logger.info(f'Выбрано видео: {video_path[0]}')
            self.video_manager.video_path = video_path[0]
            self.btn_start.setVisible(True)
            self.flags_current_operation = FLAG_OPERATION.VIDEO
        else:
            self.logger.info('Не выбрано видео или отмена')

    def open_catalog_images(self):
        self.flags_current_operation = FLAG_OPERATION.NO_SELECT
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder_path:
            return
        image_paths = load_images_from_folder(folder_path)
        if not image_paths:
            self.logger.info("В выбранной папке нет изображений.")
            self.btn_start.setVisible(False)

            return
        else:
            self.logger.info(f' В выбранной папке найдено: {len(image_paths)} изображений.'
                             f' \n Нажмите кнопку "Запустить" ')
            self.btn_start.setVisible(True)
            self.folders_manager.images_path = image_paths
            self.flags_current_operation = FLAG_OPERATION.FOLDER

    def click_checkbox_folders(self):
        self.logger.info('Выбран режим выбора каталога с фото.')
        self.camera_manager.disconnect_cam()
        self.btn_open_video.setVisible(False)
        self.btn_open_folder.setVisible(True)
        self.btn_start.setVisible(False)
        self.btn_stop.setVisible(False)

    def click_checkbox_video_file(self):
        self.logger.info('Выбран режим выбора видеофайла')
        self.camera_manager.disconnect_cam()
        self.folders_manager.stop_stream()
        self.btn_open_folder.setVisible(False)
        self.btn_start.setVisible(False)
        self.btn_open_video.setVisible(True)
        self.btn_stop.setVisible(False)

    def click_checkbox_webcam(self):
        self.logger.info('Выбран режим показа с вебкамеры [0]')
        self.folders_manager.stop_stream()
        self.btn_open_video.setVisible(False)
        self.btn_open_folder.setVisible(False)
        self.btn_start.setVisible(True)
        self.flags_current_operation = FLAG_OPERATION.WEBCAM
        self.btn_stop.setVisible(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = VideoWidget()
    w.show()
    sys.exit(app.exec())
