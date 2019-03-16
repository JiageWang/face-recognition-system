import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon
from src.MainWindow import Ui_MainWindow
from src.cvthread import CVThead
from src.batchRecord import BatchRecordThead
from src.record import RecordForm
import pickle

import sys


class MainWindows(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindows, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('./icon/face.png'))
        self.setWindowTitle('人脸识别系统')
        # self.setFixedSize(self.width(), self.height())

        self.frame = None
        self.pixmap = None
        self.cap_open_flag = False

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # OpenCV
        self.cv_thread = CVThead()
        self.cv_thread.sin_out_pixmap.connect(self.process_signal)
        self.cv_thread.sin_out_frame.connect(self.process_signal)
        self.cv_thread.sin_out_names.connect(self.process_signal)

        # 参数调节
        self.radioButton1.setChecked(True)
        self.radioButton1.toggled.connect(self.upsample_state)
        self.radioButton2.setChecked(False)
        self.radioButton2.toggled.connect(self.upsample_state)
        self.radioButton4.setChecked(False)
        self.radioButton4.toggled.connect(self.upsample_state)

        # self.radioButtonCNN.setChecked(True)
        # self.radioButtonCNN.toggled.connect(self.change_model)
        # self.radioButtonHOG.setChecked(False)
        # self.radioButtonHOG.toggled.connect(self.change_model)

        self.valueLabel.setText(str(self.cv_thread.tolerance))
        self.toleranceSlider.setMinimum(0)
        self.toleranceSlider.setMaximum(100)
        self.toleranceSlider.setSingleStep(1)
        self.toleranceSlider.setValue(self.cv_thread.tolerance)
        self.toleranceSlider.valueChanged.connect(self.tolerance_state)

        # 数据库
        self.registerButton.setEnabled(False)
        self.registerButton.clicked.connect(self.record)
        self.batchRecordThead = BatchRecordThead()
        self.batchRecordThead.sin_out_tuple.connect(self.batch_record)
        self.batchRegisterButton.clicked.connect(self.batch_record)
        self.loadInButton.clicked.connect(self.load_in)
        self.loadOutButton.clicked.connect(self.load_out)

        # 相机设置
        self.CameraCheckBox.stateChanged.connect(self.use_external_camera)
        self.CameraButton.toggled.connect(self.start_camera)
        self.CameraButton.setCheckable(True)

    def load_out(self):
        with open('encodings.pkl', 'wb') as f:
            pickle.dump(self.cv_thread.known_face_encodings, f)
        with open('names.pkl', 'wb') as f:
            pickle.dump(self.cv_thread.known_face_names, f)

    def load_in(self, ):
        try:
            with open('encodings.pkl', 'rb') as f:
                self.cv_thread.known_face_encodings = pickle.load(f)
            with open('names.pkl', 'rb') as f:
                self.cv_thread.known_face_names = pickle.load(f)
        except IOError:
            print("Files not found")

    def change_model(self):
        pass

    # 处理信号
    def process_signal(self, signal):
        if isinstance(signal, QPixmap):
            self.pixmap = signal
        elif isinstance(signal, list):
            if self.cap_open_flag:
                self.infoLabel.setText("当前人数：{0}".format(len(signal)))
                self.nameBrowser.setText('\n'.join(signal))
        elif isinstance(signal, np.ndarray):
            self.frame = signal

    # 是否使用外接摄像头
    def use_external_camera(self, status):
        if status == Qt.Checked:
            self.cv_thread.capID = 1
            self.cv_thread.change_cap()
        else:
            self.cv_thread.capID = 0
            self.cv_thread.change_cap()

    # 打开/关闭摄像头
    def start_camera(self, status):
        if status:
            self.cap_open_flag = True
            self.cv_thread.start()
            self.CameraButton.setText('关闭摄像头')
            self.registerButton.setEnabled(True)
            self.timer.start()
        else:
            self.cap_open_flag = False
            self.cv_thread.quit()
            if self.timer.isActive():
                self.timer.stop()
            self.CameraButton.setText('打开摄像头')
            self.registerButton.setEnabled(False)
            self.FrameLabel.clear()
            self.infoLabel.setText("当前人员")
            self.nameBrowser.clear()

    # 更新画面
    def update_frame(self):
        if self.cap_open_flag and self.pixmap is not None:
            self.FrameLabel.setPixmap(self.pixmap)
            self.FrameLabel.setScaledContents(True)
        else:
            self.FrameLabel.clear()

    # 修改up_sample
    def upsample_state(self):
        if self.radioButton1.isChecked():
            self.cv_thread.up_sample = 1
        elif self.radioButton2.isChecked():
            self.cv_thread.up_sample = 2
        elif self.radioButton4.isChecked():
            self.cv_thread.up_sample = 4

    # 修改tolerance
    def tolerance_state(self):
        self.cv_thread.tolerance = self.toleranceSlider.value()
        self.valueLabel.setText(str(self.cv_thread.tolerance))

    # 用户注册
    def record(self):
        # 用户信息填写对话框
        if self.frame is not None:
            dialog_window = RecordForm(self.frame.copy())
            dialog_window.exec_()
            if dialog_window.name is not None:
                self.cv_thread.known_face_encodings.extend(dialog_window.encoding)
                self.cv_thread.known_face_names.append(dialog_window.name)
            del dialog_window
        else:
            QMessageBox.question(self, 'Warning', "未检测到人脸", QMessageBox.Yes, QMessageBox.Yes)

    # 通过照片路径注册
    def batch_record(self, signal):
        if isinstance(signal, tuple):
            self.cv_thread.known_face_names.extend(signal[0])
            self.cv_thread.known_face_encodings.extend(signal[1])
            self.batchRecordThead.quit()
        else:
            self.batchRecordThead.start()

    def save_data(self, list1, list2):
        self.known_face_encodings = list2
        self.known_face_names = list1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.keys()[2])
    window = MainWindows()
    window.show()
    sys.exit(app.exec_())
