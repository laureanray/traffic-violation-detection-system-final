# Imports
from widgets import Detection, Config
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import application as app


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        # Load GUI
        uic.loadUi('user_interface/MainWindow.ui', self)
        self.detection = Detection()
        self.configs = Config()
        self.startButton.clicked.connect(self.onStartButton)
        self.actionConfigs.triggered.connect(self.onTriggeredConfig)
        self.detection.show()

    def onStartButton(self):
        print('Start')
        if app.State.isStarted:
            self.startButton.setText("Start")
            app.State.isStarted = False
            self.detection.close()
        else:
            self.startButton.setText("Stop")
            app.State.isStarted = True
            self.detection.show()

    def onTriggeredConfig(self):
        print('Configs')
        self.configs.show()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.setWindowTitle('Main')
    # window.show()
    sys.exit(app.exec_())
