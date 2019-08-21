import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QCheckBox, QPushButton, QVBoxLayout, QFrame, QSplitter, \
    QHBoxLayout
from application import Details, State
from widgets.dashboard import Dashboard
from widgets.logging import Logging

import cv2 as cv

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = Details.application_name
        self.left = 10
        self.top = 10
        self.width = Details.width
        self.height = Details.height
        self.cap = cv.VideoCapture(0)
        # Instantiate the logging widget

        self.initUI()

    def initUI(self):
        # Layout Manager

        # Instantiate the dashboard widget class
        State.logging = Logging()
        State.dashboard = Dashboard()


        # Initialize the panels
        self.topPanel = QFrame(self)
        self.topPanel.setFrameShape(QFrame.StyledPanel)

        self.bottomPanel = QFrame(self)
        self.bottomPanel.setFrameShape(QFrame.StyledPanel)

        # Set the layout manager of the panels
        self.topPanelLayout = QHBoxLayout()
        self.topPanelLayout.addWidget(State.dashboard)
        self.topPanel.setLayout(self.topPanelLayout)

        self.bottomPanelLayout = QHBoxLayout()
        self.bottomPanelLayout.addWidget(State.logging)
        self.bottomPanel.setLayout(self.bottomPanelLayout)
        self.bottomPanel.setFixedHeight(220)

        # Initialize the splitter
        verticalSpliter = QSplitter(Qt.Vertical)
        verticalSpliter.addWidget(self.topPanel)
        verticalSpliter.addWidget(self.bottomPanel)


        # Set the central widget
        self.setCentralWidget(verticalSpliter)
        self.initMenuBar()
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # If we get here no errors
        State.logging.log("Application", "Main GUI started successfully")
        self.show()


    def initMenuBar(self):
        # Menubar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        aboutMenu = mainMenu.addMenu('About')
        viewMenu = mainMenu.addMenu('View')

        exitAction = QtWidgets.QAction('Exit', self)
        viewLogAction = QtWidgets.QAction('Log', self, checkable=True)
        viewLogAction.setChecked(True)

        viewMenu.addAction(viewLogAction)
        fileMenu.addAction(exitAction)
        exitAction.triggered.connect(self.close)
        viewLogAction.triggered.connect(self.toggleLogWindow)


    def toggleLogWindow(self, state):
        if state:
            State.logging.show()
            self.bottomPanel.setFixedHeight(220)
        else:
            State.logging.close()
            self.bottomPanel.setFixedHeight(0)







if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyle("GTK+")

State.main = Main()
sys.exit(app.exec_())