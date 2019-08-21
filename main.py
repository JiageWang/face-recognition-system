import os
import time

import cv2
import sys
import numpy as np
from PyQt5.Qt import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from libs import FaceModel
from libs.utils.utils import show_bboxes, new_excel, compare_embedding
from windwos.record_window import RecordForm
from ui.main_window import Ui_MainWindow


class FaceRecordSystem(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(FaceRecordSystem, self).__init__()
        self.setupUi(self)
        # self.setWindowIcon(QIcon(iconMain))
        self.setWindowTitle('人脸识别系统')

        # 设置定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(27)

        # 设置按键
        self.network_camera_radio.toggled.connect(self.change_camera)
        self.local_camera_radio.toggled.connect(self.change_camera)
        self.confirm_button.clicked.connect(self.confirm_change_camera)
        self.record_button.clicked.connect(self.record_staff)
        self.imgrecord_button.clicked.connect(self.img_record)

        # 设置相机
        self.cap = cv2.VideoCapture(0)
        self.use_local_camera = True

        # 人脸检测模型
        self.face_model = FaceModel(None)

        # 类内变量
        self.names = []
        self.embeddings = []
        self.face = None

        # 初始化表格
        self.init_table()

    def clear_table(self):
        self.table.clear()
        self.init_table()

    def record_staff(self):
        # 用户信息填写对话框
        if self.face is not None:
            face_roi = self.face.copy()
            dialog_window = RecordForm(face_roi)
            dialog_window.exec_()
            if dialog_window.accepted:
                self.names.append(dialog_window.name)
                self.embeddings.append(self.face_model.arcface(face_roi))
        else:
            QMessageBox.question(self, 'Warning', "未检测到人脸", QMessageBox.Yes, QMessageBox.Yes)

    def img_record(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, '人脸照片', './', 'Image files(*.jpg *.png)')
        print(file_names)
        if not file_names:
            return
        for file_name in file_names:
            name = os.path.basename(file_name).split('.')[0]
            img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
            landmarks, bboxs, faces, embeddings = self.face_model(img)
            embedding = embeddings[0]
            self.names.append(name)
            self.embeddings.append(embedding)

    def change_camera(self):
        if self.local_camera_radio.isChecked():
            self.rtsp_editline.setEnabled(False)
            self.confirm_button.setEnabled(False)
            if not self.use_local_camera:
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
            self.use_local_camera = True

        if self.network_camera_radio.isChecked():
            self.rtsp_editline.setEnabled(True)
            self.confirm_button.setEnabled(True)
            self.cap.release()
            self.use_local_camera = False

    def confirm_change_camera(self):
        path = self.rtsp_editline.text()
        self.cap = cv2.VideoCapture(path)

    def init_table(self):
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(['姓名'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setStyleSheet('color:rgb(100,100,100,250) ;font-size:15px')
        self.table.setStyleSheet('font-size:15px')
        self.table.setAlternatingRowColors(True)

    def update_table(self, names):
        self.table.setRowCount(len(names))
        # self.table.setModel(QStandardItemModel(len(names), 2))
        for i, name in enumerate(names):
            self.table.setItem(i, 0, QTableWidgetItem(name))
        self.table.show()

    def update(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            self.frame = frame
            landmarks, bboxs, faces, embeddings = self.face_model(frame)
            if len(faces) > 0:
                self.face = faces[0]
            names = []
            for embedding in embeddings:
                idx = compare_embedding(embedding, self.embeddings)
                names.append(self.names[idx] if idx is not None else 'unknown')
            self.update_table(names)

            frame = show_bboxes(frame, bboxs, landmarks, names)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, c = frame.shape  # 获取图片形状
            image = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.frame_label.setPixmap(pixmap)
            self.frame_label.setScaledContents(True)  # 设置图片随QLabel大小缩放
        else:
            self.frame_label.clear()
            self.face = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.keys()[2])
    window = FaceRecordSystem()
    window.show()
    sys.exit(app.exec_())
