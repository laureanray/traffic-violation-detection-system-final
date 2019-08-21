import cv2 as cv
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QFileDialog, QComboBox, QFrame, \
    QLineEdit

from application import State
from widgets.logging import Logging


class Dashboard(QWidget):
    def __init__(self, parent=None):
        super(Dashboard, self).__init__(parent)
        # Nested layouts
        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()

        label = QLabel(self)
        label.setText('Model: ')
        label.setFixedWidth(50)

        button = QPushButton('Load model')
        button.setFixedWidth(150)
        button.clicked.connect(self.openModelDialog)

        label2 = QLabel(self)
        label2.setText('Pbtext: ')
        label2.setFixedWidth(50)

        button2 = QPushButton('Load .pbtext')
        button2.setFixedWidth(150)
        button2.clicked.connect(self.openPbTextDialog)

        # Selector
        self.selector = QComboBox(self)
        self.selector.addItem("Load footage from file")
        self.selector.addItem("Load footage from camera")
        self.selector.addItem("Load footage from ip address")
        self.selector.setFixedWidth(250)
        self.selector.activated[str].connect(self.onComboBoxChange)

        # Load footage from file frame and layout
        widthInput = QLineEdit(self)
        heightInput = QLineEdit(self)
        self.footageNameLabel = QLabel()
        widthLabel = QLabel('Width')
        heightLabel = QLabel('Height')
        inputLayout = QHBoxLayout()
        inputLayout.addWidget(widthLabel)
        inputLayout.addWidget(widthLabel)
        inputLayout.addWidget(widthInput)
        inputLayout.addWidget(heightLabel)
        inputLayout.addWidget(heightInput)

        self.loadFootageFrame = QFrame()
        loadFootageLayout = QHBoxLayout()
        loadFootageButton = QPushButton("Load Footage")
        # Add click listener
        loadFootageButton.clicked.connect(self.openFootageDialog)
        loadFootageLayout.addWidget(loadFootageButton)
        loadFootageLayout.addLayout(inputLayout)
        self.loadFootageFrame.setLayout(loadFootageLayout)

        loadFootageLayout.setContentsMargins(0,0,0,0)

        # Load from camera
        self.loadCameraFrame = QFrame()
        loadCameraLayout = QHBoxLayout()
        loadCameraLayout.addWidget(QPushButton("Load camera"))
        self.loadCameraFrame.setLayout(loadCameraLayout)
        loadCameraLayout.setContentsMargins(0,0,0,0)

        # Load from IP Address
        self.loadIPFrame = QFrame()
        loadIPLayout = QHBoxLayout()
        loadIPLayout.addWidget(QPushButton("Load IP"))
        self.loadIPFrame.setLayout(loadIPLayout)
        loadIPLayout.setContentsMargins(0,0,0,0)

        # Initialize state
        self.loadCameraFrame.setVisible(False)
        self.loadIPFrame.setVisible(False)

        # Add widgets to HBOX
        self.hbox.setAlignment(Qt.AlignLeft)
        self.hbox.addWidget(label)
        self.hbox.addWidget(button)
        self.hbox.addWidget(label2)
        self.hbox.addWidget(button2)


        # Set the Vbox
        self.vbox.setAlignment(Qt.AlignTop)
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.selector)
        self.vbox.addWidget(self.loadFootageFrame)
        self.vbox.addWidget(self.loadIPFrame)
        self.vbox.addWidget(self.loadCameraFrame)


        self.setLayout(self.vbox)
        self.setGeometry(10, 10, 50, 20)
        # self.show()

    def openModelDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Load model", "",
                                                  "Load pb (*.pb), All Files (*.*)", options=options)
        if fileName:
            print(fileName)
            State.model_path = fileName
            State.logging.log("File", fileName)

    def openPbTextDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Load model", "",
                                                  "Load pb (*.pb), All Files (*.*)", options=options)
        if fileName:
            State.pbtext_path = fileName
            State.logging.log("File", fileName)

    def openFootageDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Load footage", "",
                                                  "Video file (*.mp4), All Files (*.*)", options=options)
        if fileName:
            self.footageNameLabel.setText(fileName)
            State.footage = fileName
            State.logging.log("File", fileName)


    def addWidgetToVBox(self, widget):
        if widget is not None:
            self.vbox.addWidget(widget)

    def removeWidgetFromVBox(self, widget):
        if widget is not None:
            self.vbox.removeWidget(widget)

    def onComboBoxChange(self, text):
        self.loadFootageFrame.setVisible(False)
        self.loadCameraFrame.setVisible(False)
        self.loadIPFrame.setVisible(False)
        if text == "Load footage from file":
            self.loadFootageFrame.setVisible(True)
        elif text == "Load footage from camera":
            self.loadCameraFrame.setVisible(True)
        else:
            self.loadIPFrame.setVisible(True)
        State.logging.log("Dashboard", text)

