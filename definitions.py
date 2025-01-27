import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
SAVE_PATH = os.path.join(ROOT_DIR,'save_result')
LOG_PATH = os.path.join(ROOT_DIR,'logs')
SAVE_PATH_WEBCAM = os.path.join(SAVE_PATH, 'webcam')
SAVE_PATH_FOLDER = os.path.join(SAVE_PATH, 'photos')
SAVE_PATH_VIDEO = os.path.join(SAVE_PATH, 'video')