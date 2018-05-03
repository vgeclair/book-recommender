# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rate.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_RateWidget(object):
    def setupUi(self, RateWidget):
        RateWidget.setObjectName("RateWidget")
        RateWidget.resize(161, 290)
        RateWidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.graphicsView_cover = QtWidgets.QGraphicsView(RateWidget)
        self.graphicsView_cover.setGeometry(QtCore.QRect(30, 20, 98, 146))
        self.graphicsView_cover.setObjectName("graphicsView_cover")
        self.label_title = QtWidgets.QLabel(RateWidget)
        self.label_title.setGeometry(QtCore.QRect(0, 190, 161, 61))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_title.setFont(font)
        self.label_title.setAcceptDrops(False)
        self.label_title.setTextFormat(QtCore.Qt.AutoText)
        self.label_title.setScaledContents(False)
        self.label_title.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_title.setWordWrap(True)
        self.label_title.setObjectName("label_title")
        self.label_rating = QtWidgets.QLabel(RateWidget)
        self.label_rating.setGeometry(QtCore.QRect(10, 260, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_rating.setFont(font)
        self.label_rating.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_rating.setObjectName("label_rating")
        self.spinBox_rating = QtWidgets.QSpinBox(RateWidget)
        self.spinBox_rating.setGeometry(QtCore.QRect(100, 260, 42, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.spinBox_rating.setFont(font)
        self.spinBox_rating.setMaximum(5)
        self.spinBox_rating.setObjectName("spinBox_rating")

        self.retranslateUi(RateWidget)
        QtCore.QMetaObject.connectSlotsByName(RateWidget)

    def retranslateUi(self, RateWidget):
        _translate = QtCore.QCoreApplication.translate
        RateWidget.setWindowTitle(_translate("RateWidget", "Form"))
        self.label_title.setText(_translate("RateWidget", "This is a very long name of a book that might not even fit"))
        self.label_rating.setText(_translate("RateWidget", "Your rating:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    RateWidget = QtWidgets.QWidget()
    ui = Ui_RateWidget()
    ui.setupUi(RateWidget)
    RateWidget.show()
    sys.exit(app.exec_())

