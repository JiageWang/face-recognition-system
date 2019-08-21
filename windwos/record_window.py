from PyQt5.QtGui import QImage, QRegExpValidator, QPixmap, QIcon
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import *
from ui.record_ui import Ui_Record
import cv2


class RecordForm(QDialog, Ui_Record):
    def __init__(self, face_iou):
        super(RecordForm, self).__init__()
        self.setupUi(self)
        # self.setWindowIcon(QIcon(iconRecord))
        self.setWindowTitle('人脸信息采集')
        self.face_iou = face_iou
        self.accepted = False

        # 显示图片
        qformat = QImage.Format_RGB888
        face_iou = cv2.resize(face_iou, (self.FrameLabel.height(), self.FrameLabel.width()))
        face_iou = cv2.cvtColor(face_iou, cv2.COLOR_BGR2RGB)
        image = QImage(face_iou, face_iou.shape[1], face_iou.shape[0], face_iou.strides[0], qformat)
        self.FrameLabel.setPixmap(QPixmap.fromImage(image))
        self.FrameLabel.setScaledContents(True)

        # 正则表达式限制输入
        name_regx = QRegExp('^[\u4e00-\u9fa5A-Za-z]{1,10}$')
        name_validator = QRegExpValidator(name_regx, self.NameLineEdit)
        self.NameLineEdit.setValidator(name_validator)

        # 判断是否保存注册
        self.DialogBox.accepted.connect(self.dialog_box_accept)
        self.DialogBox.rejected.connect(self.dialog_box_reject)

    def dialog_box_accept(self):
        self.accepted = True
        self.name = self.NameLineEdit.text()
        self.close()

    def dialog_box_reject(self):
        self.accepted = False
        self.close()

    def closeEvent(self, event):
        self.close()
