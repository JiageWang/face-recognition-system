from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from face_recognition import face_locations, compare_faces, face_encodings
from .util import put_chinese
import cv2
import logging
import numpy as np


class CVThead(QThread):
    '''多线程处理图像'''
    sin_out_names = pyqtSignal(list)
    sin_out_frame = pyqtSignal(np.ndarray)
    sin_out_pixmap = pyqtSignal(QPixmap)  # 自定义信号，执行run()函数时，从相关线程发射此信号

    def __init__(self):
        super(CVThead, self).__init__()
        self.capID = 0
        self.cap_list = [cv2.VideoCapture(0), cv2.VideoCapture(1)]
        self.cap_used = None
        self.ratio = 0.25
        self.model = 'cnn'
        self.tolerance = 42
        self.up_sample = 1
        self.jitters = 10
        self.face = None
        self.face_num = 0
        self.encoding = None
        self.known_face_encodings = []
        self.known_face_names = []

    def __del__(self):
        self.wait()

    def change_cap(self):
        self.cap_used = self.cap_list[self.capID]

    def work(self):
        # 获取摄像图像进行人脸识别
        ret, origin_frame = self.cap_used.read()
        # print(origin_frame.shape)
        # print(origin_frame.shape)

        if not ret:
            logging.debug("cap not open")
            return
        else:
            frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)

            # 检测人脸
            locations = face_locations(small_frame,
                                       model=self.model,
                                       number_of_times_to_upsample=self.up_sample)

            # 生成人脸向量
            encodings = face_encodings(small_frame,
                                       locations,
                                       num_jitters=self.jitters)

            names = []
            for encoding in encodings:
                name = ""
                # 人脸向量匹配
                matches = compare_faces(self.known_face_encodings,
                                        encoding,
                                        tolerance=self.tolerance/100)
                # 如果匹配则使用第一个匹配的人名
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                names.append(name)

            # 显示人脸位置与人名标注
            for (top, right, bottom, left), name in zip(locations, names):
                # 放大回原图大小
                top, right, bottom, left = (x*4 for x in (top, right, bottom, left))
                # 显示方框
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                if name != "":
                    cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    # 显示人名
                    frame = put_chinese(frame, (left + 10, bottom), name, (255, 255, 255), 25)
            # 转换成QImage
            qformat = QImage.Format_RGB888
            out_image = QImage(frame,
                               frame.shape[1],
                               frame.shape[0],
                               frame.strides[0],
                               qformat)
            # 输出信号
            self.sin_out_names.emit(names)
            self.sin_out_frame.emit(origin_frame.copy())
            self.sin_out_pixmap.emit(QPixmap.fromImage(out_image).copy())  # 传出副本，不然会出现GBR

    def run(self):
        self.cap_used = self.cap_list[self.capID]
        while True:
            self.work()








