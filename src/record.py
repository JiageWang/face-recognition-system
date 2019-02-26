from PyQt5.QtGui import QImage,QRegExpValidator, QPixmap, QIcon
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import *
from .recordUI import Ui_Record
from face_recognition import face_encodings, face_locations, face_landmarks
import cv2


class RecordForm(QDialog, Ui_Record):
    def __init__(self, frame):
        super(RecordForm, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('./icon/record.png'))
        self.setWindowTitle('人脸信息采集')
        self.location = None
        self.encoding = None
        self.name = None
        self.face_img = None
        self.frame = frame

        # 人脸识别
        try:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.location = face_locations(self.frame, 1, 'cnn')
            if len(self.location) == 1:
                self.encoding = face_encodings(self.frame, self.location, 10)
                self.marks = face_landmarks(self.frame, self.location, 'large')

                # 标记关键点
                for person in self.marks:
                    for points, positions in person.items():
                        for position in positions:

                            cv2.circle(self.frame, position, 2, (0, 255, 0), thickness=2)
                top, right, bottom, left = self.location[0]
                self.face_img = self.frame[top-35:bottom+35, left-35:right+35, :]
        except IndexError:
            QMessageBox.question(self, 'Warning', "未检测到关键点",
                                 QMessageBox.Yes, QMessageBox.Yes)

        # 显示图片
        qformat = QImage.Format_RGB888
        if self.face_img is None:
            self.close()
        else:
            self.face_img = cv2.resize(self.face_img,
                                       (self.FrameLabel.height(), self.FrameLabel.width()))
            out_image = QImage(self.face_img,
                               self.face_img.shape[1],
                               self.face_img.shape[0],
                               self.face_img.strides[0],
                               qformat)
            self.FrameLabel.setPixmap(QPixmap.fromImage(out_image))
            self.FrameLabel.setScaledContents(True)


        # 正则表达式限制输入
        name_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        name_validator = QRegExpValidator(name_regx, self.NameLineEdit)
        self.NameLineEdit.setValidator(name_validator)
        self.NameLineEdit.setText("请输入中文名")

        # 判断是否保存注册
        self.DialogBox.accepted.connect(self.dialog_box_accept)
        self.DialogBox.rejected.connect(self.dialog_box_reject)

    def dialog_box_accept(self):
        self.name = self.NameLineEdit.text()
        self.close()

    def dialog_box_reject(self):
        self.close()

    def closeEvent(self, event):
        self.close()






