# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'videoPlayergWrHMi.ui'
##
# Created by: Qt User Interface Compiler version 6.4.3
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


import sys

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt,
                            QStringListModel, QThread, Signal, Slot)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform,
                           QImage, QPixmap)
from PySide6.QtWidgets import (QApplication, QDialog, QGraphicsView, QListView,
                               QPushButton, QSizePolicy, QWidget, QLabel, QVBoxLayout)

from PySide6.QtMultimediaWidgets import QVideoWidget

from PySide6.QtMultimedia import (QMediaPlayer)

class VideoThread(QThread):
    update_frame = Signal(np.ndarray)

    def __init__(self, file_path: str, width: int, height: int, num_frames: int, fps: int):
        super().__init__()

        self.file_path = file_path
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.threadPaused = False
        self.stop_flag = False
        self.stop_event = False

    def run(self):
        # open the video file for reading
        with open(self.file_path, "rb") as file:
            # read each frame of the video and display it
            for i in range(self.num_frames):

                while self.threadPaused:
                    # print("timeSleep")
                    time.sleep(0.1)

                if self.stop_event:
                    print("break")
                    break

                # read the raw pixel data for the current frame
                raw_data = file.read(self.width * self.height * 3)

                # convert the raw data to a numpy array of pixel values
                pixels = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 3))

                # convert the RGB pixels to BGR format for displaying in OpenCV
                bgr_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                # emit a signal with the current frame data
                self.update_frame.emit(bgr_image)

                # wait for the specified number of milliseconds before showing the next frame
                time.sleep(1 / self.fps)

    def stop(self):
        #set the signal to break for loop
        self.stop_event = True

        #kill the thread and emit the finish signal
        self.quit()
        self.finished.emit()
    
    def pause(self):
        self.threadPaused = True

    def resum(self):
        self.threadPaused = False


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # create a label widget to display the video frames
        self.label = QLabel()

        # create a layout for the widget and add the label to it
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        # convert the frame data to a QImage for display in the label widget
        qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # display the frame in the label widget
        self.label.setPixmap(pixmap)

class Ui_Dialog(object):

    def __init__(self):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(738, 447)
        self.btnPlay = QPushButton(Dialog)
        self.btnPlay.setObjectName(u"btnPlay")
        self.btnPlay.setGeometry(QRect(430, 380, 93, 28))
        self.btnPlay.setStyleSheet("background-color: grey;")

        self.btnPause = QPushButton(Dialog)
        self.btnPause.setObjectName(u"btnPause")
        self.btnPause.setGeometry(QRect(530, 380, 93, 28))
        self.btnPause.setStyleSheet("background-color: grey;")

        self.btnStop = QPushButton(Dialog)
        self.btnStop.setObjectName(u"btnStop")
        self.btnStop.setGeometry(QRect(630, 380, 93, 28))
        self.btnStop.setStyleSheet("background-color: grey;")

        self.listView = QListView(Dialog)
        self.listView.setObjectName(u"listView")
        self.listView.setGeometry(QRect(30, 40, 256, 361))   
        
        #video player button attributes
        self.start = False
        self.paused = False

        # create a label widget to display the video frames
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"video_widget")
        self.label.setGeometry(QRect(350, 40, 361, 291))

        # create a layout for the widget and add the label to it
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        #self.setLayout(layout)
        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)

        self.btnPlay.clicked.connect(self.play_video)
        self.btnPause.clicked.connect(self.pause_video)
        self.btnStop.clicked.connect(self.stop_video)

        # Add some strings to the model
        strings = ["Item 1", "Item 2", "Item 3"]
        self.model = QStringListModel()
        self.model.setStringList(strings)

        # Set the model on the list view
        self.listView.setModel(self.model)

        # close the OpenCV window
        cv2.destroyAllWindows()


    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        # convert the frame data to a QImage for display in the label widget
        qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # display the frame in the label widget
        self.label.setPixmap(pixmap)

    def play_video(self):
        print("play_video")

        if not self.start:
            # define the path to the RGB video file and the video parameters
            file_path = Path("./InputVideo.rgb")
            width, height = 480, 270
            fps, num_frames = 30, 8682

            # create a thread to read the video and emit signals with the frame data
            self.video_thread = VideoThread(file_path, width, height, num_frames, fps)
            self.video_thread.update_frame.connect(self.update_frame)

            # start the thread to read the video
            self.video_thread.start()
            self.start = True

        if self.start:
            if self.paused:
                self.video_thread.resum()
                self.paused = False
                self.btnPlay.setText(
                QCoreApplication.translate("Dialog", u"Play", None))

    def pause_video(self):
        #set paused to True 
        self.video_thread.pause()
        if not self.paused:
            self.paused = True
            self.btnPlay.setText(
            QCoreApplication.translate("Dialog", u"Resum", None))


    def stop_video(self):
        # call stop function in the thread class
        self.video_thread.stop()

        # reset the all related var
        self.start = False
        self.paused = False


    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(
            QCoreApplication.translate("Dialog", u"Dialog", None))
        self.btnPlay.setText(
            QCoreApplication.translate("Dialog", u"Play", None))
        self.btnPause.setText(
            QCoreApplication.translate("Dialog", u"Pause", None))
        self.btnStop.setText(
            QCoreApplication.translate("Dialog", u"Stop", None))
    # retranslateUi


if __name__ == '__main__':

    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    Dialog = QDialog()
    ui = Ui_Dialog()
    #ui.setupUi(Dialog)
    Dialog.show()
    # Start the event loop
    app.exec()

    # Exit the application
    sys.exit()
