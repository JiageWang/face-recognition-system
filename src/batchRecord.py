from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from .util import get_encodings


class BatchRecordThead(QThread):
    sin_out_tuple = pyqtSignal(tuple)

    def __init__(self):
        super(BatchRecordThead, self).__init__()

    def run(self):
        path = QFileDialog.getExistingDirectory()
        if path == '':
            return
        known_face_names, known_face_encodings = get_encodings(path)
        self.sin_out_tuple.emit((known_face_names, known_face_encodings))
