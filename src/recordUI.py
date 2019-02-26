# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'record.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Record(object):
    def setupUi(self, Record):
        Record.setObjectName("Record")
        Record.resize(358, 434)
        self.groupBox = QtWidgets.QGroupBox(Record)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 371, 441))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.FrameLabel = QtWidgets.QLabel(self.groupBox)
        self.FrameLabel.setGeometry(QtCore.QRect(0, -10, 361, 361))
        self.FrameLabel.setStyleSheet("background:black;")
        self.FrameLabel.setObjectName("FrameLabel")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(80, 360, 201, 41))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.NameLineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.NameLineEdit.setObjectName("NameLineEdit")
        self.gridLayout.addWidget(self.NameLineEdit, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.DialogBox = QtWidgets.QDialogButtonBox(self.groupBox)
        self.DialogBox.setGeometry(QtCore.QRect(100, 410, 156, 23))
        self.DialogBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.DialogBox.setObjectName("DialogBox")

        self.retranslateUi(Record)
        QtCore.QMetaObject.connectSlotsByName(Record)

    def retranslateUi(self, Record):
        _translate = QtCore.QCoreApplication.translate
        Record.setWindowTitle(_translate("Record", "Form"))
        self.FrameLabel.setText(_translate("Record", "TextLabel"))
        self.label_3.setText(_translate("Record", "姓名"))

