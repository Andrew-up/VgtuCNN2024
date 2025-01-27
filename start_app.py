import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import VideoWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = VideoWidget()
    w.show()
    sys.exit(app.exec())
