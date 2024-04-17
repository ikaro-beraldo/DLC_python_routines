# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Behaviour.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from main_batch_function import main_batch_run


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(313, 348)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(6, 10, 300, 131))
        self.tabWidget.setObjectName("tabWidget")
        self.BarnesMaze = QtWidgets.QWidget()
        self.BarnesMaze.setObjectName("BarnesMaze")
        self.barnespushButton = QtWidgets.QPushButton(self.BarnesMaze)  # Create the BarnesPushButton
        self.barnespushButton.clicked.connect(self.click_barnes_maze)   # Create a click action
        self.barnespushButton.setGeometry(QtCore.QRect(10, 80, 75, 23))
        self.barnespushButton.setObjectName("barnespushButton")
        self.tabWidget.addTab(self.BarnesMaze, "")
        self.OLR = QtWidgets.QWidget()
        self.OLR.setObjectName("OLR")
        self.tabWidget.addTab(self.OLR, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.barnespushButton.setText(_translate("Dialog", "Run Batch"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.BarnesMaze), _translate("Dialog", "BarnesMaze"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.OLR), _translate("Dialog", "OLR"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "OF"))

    # Barnes Maze action
    def click_barnes_maze(self):
        main_batch_run()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
