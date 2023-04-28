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

import pyaudio
import wave

import numpy as np
import cv2
import time

import re

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
                               QPushButton, QSizePolicy, QWidget, QLabel, QVBoxLayout, QMessageBox)

from PySide6.QtMultimediaWidgets import QVideoWidget

from PySide6.QtMultimedia import (QMediaPlayer)

#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#from tensorflow.keras.preprocessing import image

from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity


SHOT_THRESHOLD = 20
SAME_SHOT_WINDOWS = 10
SCENE_THRESHOLD = 0.65

# Function to extract color histogram features from an image


def extract_color_histogram_features(img, bins=8):
    # if len(img.shape) == 2 or img.shape[2] == 1:  # If the image is grayscale
    #    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    hist = cv2.calcHist([img], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to extract HOG features from an image


def extract_hog_features(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    # gray_img=img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_img, orientations=9,
                       pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
    return hog_features


# Calculate Absolute Difference
def meanAbsDiff(img1: np.ndarray, img2: np.ndarray) -> float:
    # Calculate absolute difference betweeb two continuous frames
    abs_diff = cv2.absdiff(img1, img2)

    # Calculate mean of absolute difference
    mad = np.mean(abs_diff)
    return mad


class VideoThread(QThread):
    update_frame = Signal(np.ndarray)

    def __init__(self, file_path: str, width: int, height: int, num_frames: int, fps: int, startIdx: int, index_frames: list, listView: QListView):
        super().__init__()

        self.file_path = file_path
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.threadPaused = False
        self.stop_flag = False
        self.stop_event = False
        self.start_frame_idx = startIdx
        self.pause_time = 0
        self.index_frames = index_frames
        self.listView = listView

    def run(self):
        # open the video file for reading
        with open(self.file_path, "rb") as file:

            # Set the start frame
            file.seek(self.start_frame_idx * self.width * self.height * 3)

            next_frame_time = time.monotonic()

            # current selected index on listView
            if self.listView.currentIndex().row() != -1:
                curr_idx = self.listView.currentIndex().row()
            else:
                curr_idx = 0

            # read each frame of the video and display it
            curr_frame_number = self.start_frame_idx
            while curr_frame_number < self.num_frames:

                while self.threadPaused:
                    time.sleep(0.1)

                if self.stop_event:
                    break

                # read the raw pixel data for the current frame
                raw_data = file.read(self.width * self.height * 3)

                # convert the raw data to a numpy array of pixel values
                pixels = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                # emit a signal with the current frame data
                self.update_frame.emit(pixels)

                # update the listView to the corresponding to the current frame
                # print(
                #     f'curr_frame_number: {curr_frame_number}, self.index_frames[{curr_idx}]: {self.index_frames[curr_idx]}')
                if curr_frame_number == self.index_frames[curr_idx]:
                    # Whatever it is scene->shot, shot->subshot, if they have same frame number, the last one should be selected.
                    while curr_frame_number == self.index_frames[curr_idx+1] and curr_idx < self.num_frames:
                        curr_idx += 1

                    model = self.listView.model()
                    new_idx = model.index(curr_idx)
                    self.listView.setCurrentIndex(new_idx)

                    curr_idx += 1

                # wait until it is time to display the next frame
                next_frame_time += 1 / self.fps

                time_diff = time.monotonic() - next_frame_time
                if time_diff > 0:
                    next_frame_time += time_diff

                # wait until it is time to display the next frame
                sleep_time = max(next_frame_time - time.monotonic(), 0)
                time.sleep(sleep_time)

                curr_frame_number += 1

            # End while

    def stop(self):
        # set the signal to break for loop
        self.stop_event = True

        # kill the thread and emit the finish signal
        self.quit()
        self.finished.emit()

        # refresh the video frame to white
        whites = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        whites[:, :, :] = 255
        bgr_image = cv2.cvtColor(whites, cv2.COLOR_RGB2BGR)
        self.update_frame.emit(bgr_image)

    def pause(self):
        self.threadPaused = True

    def resume(self):
        self.threadPaused = False


class AudioThread(QThread):

    def __init__(self, file_path, startIdx: int, chunk_size=1024):
        super().__init__()

        self.file_path = file_path
        self.chunk_size = chunk_size
        self.thread_paused = False
        self.stop_event = False
        self.start_frame_idx = startIdx
        self.fps = 30

    def run(self):
        # Change later
        print(self.file_path)
        wf = wave.open(str(self.file_path), 'rb')
        p = pyaudio.PyAudio()

        print(f'self.start_frame_idx: {self.start_frame_idx}')

        # set the start position of audio
        starting_time_offset = self.start_frame_idx / self.fps
        starting_frame_offset = int(starting_time_offset * wf.getframerate())
        wf.setpos(starting_frame_offset)

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=int(wf.getframerate()),
                        output=True)

        data = wf.readframes(self.chunk_size)
        audio_array = np.frombuffer(data, dtype=np.int16)

        while data != b'' and not self.stop_event:

            # when video is paused
            while self.thread_paused:
                time.sleep(0.1)

            if self.stop_event:
                break

            stream.write(data)
            data = wf.readframes(self.chunk_size)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def pause(self):
        self.thread_paused = True

    def resume(self):
        self.thread_paused = False

    def stop(self):
        self.stop_event = True

        # kill the thread and emit the finish signal
        self.quit()
        self.finished.emit()


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
        qimage = QImage(
            frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # display the frame in the label widget
        self.label.setPixmap(pixmap)


class Ui_Dialog(object):

    def __init__(self, Dialog, args):
        print("Parameters: ", args)

        # define the path to the RGB video file and the video parameters
        self.video_file_path = Path(args[1])
        self.audio_file_path = Path(args[2])
        self.width, self.height = 480, 270
        self.fps = 30
        self.num_frames = (int)(self.calNumOfFrame())

        self.shot_frame_idx = []
        self.shot_frames_gray = []
        self.shot_frames_bgr = []

        self.index_labels = []
        self.index_frames = []

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

        # video player button attributes
        self.start = False
        self.paused = False

        # create a label widget to display the video frames
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"video_widget")
        self.label.setGeometry(QRect(350, 40, 361, 291))

        # create a layout for the widget and add the label to it
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # self.setLayout(layout)
        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)

        # audio properties
        self.start_frame_idx = 0

        self.btnPlay.clicked.connect(self.play_video)
        self.btnPause.clicked.connect(self.pause_video)
        self.btnStop.clicked.connect(self.stop_video)

        # Connect the clicked signal to the custom slot
        self.listView.clicked.connect(self.on_item_clicked)

        ###################
        # Search shots
        ###################
        self.searchShots()

        ##########################
        # Extract Scene grouping shots
        ##########################
        self.extractScenes()

        # Add strings to the model
        self.model = QStringListModel()
        self.model.setStringList(self.index_labels)

        # Set the model on the list view, Prevent users from changing shot name
        self.listView.setModel(self.model)
        self.listView.setEditTriggers(QListView.NoEditTriggers)

        # close the OpenCV window
        cv2.destroyAllWindows()

    def calNumOfFrame(self):
        width, height = self.width, self.height
        num_channels = 3
        bits_per_channel = 8

        frame_size = width * height * num_channels * (bits_per_channel / 8)
        file_size = os.path.getsize(self.video_file_path)

        num_frames = file_size // frame_size

        return num_frames

    def searchShots(self):
        shot_number = 0
        # open the video file for reading
        with open(self.video_file_path, "rb") as file:
            # read each frame of the video and display it
            for i in range(self.num_frames):

                # read the raw pixel data for the current frame
                raw_data = file.read(self.width * self.height * 3)

                # convert the raw data to a numpy array of pixel values
                pixels = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                # convert the RGB scale to Grey scale
                bgr_image = pixels
                gray_image = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

                # calculate the pixel difference between two frames
                if i > 0:

                    # Calculate mean of absolute difference
                    mad = meanAbsDiff(gray_image, prev_gray_image)

                    if mad > SHOT_THRESHOLD:
                        # Print MAD value
                        print("shot_number : ", shot_number+1,
                              ", MAD: ", mad,
                              ", Frame Index: ", i)
                        if (i - self.shot_frame_idx[shot_number-1] <= SAME_SHOT_WINDOWS):
                            print(
                                "We have the same shots between same_shot_windows. Change past shot to new shot!!")
                            self.shot_frame_idx[shot_number-1] = i
                            #print("We have the same shots between same_shot_windows. Skip it!!")
                        else:
                            # Add it to the array.
                            shot_number = shot_number + 1
                            self.shot_frame_idx.append(i)
                            self.shot_frames_gray.append(gray_image)
                            self.shot_frames_bgr.append(bgr_image)

                else:
                    # For the first frame we do not calculate the pixel difference and just add it to the array.
                    shot_number = shot_number + 1
                    self.shot_frame_idx.append(i)
                    self.shot_frames_gray.append(gray_image)
                    self.shot_frames_bgr.append(bgr_image)

                # store the current frame as the previous frame for the next iteration
                prev_gray_image = gray_image.copy()

    def extractScenes(self):
        # Extract features from frames
        frame_features = np.vstack([np.hstack((extract_color_histogram_features(
            frame), extract_hog_features(frame))) for frame in self.shot_frames_bgr])

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(frame_features)

        # Group shots into scenes using a threshold
        shotNum = 1
        sceneNum = 1
        frame_idx = 0
        current_scene = [self.shot_frames_bgr[0]]

        # Save Index labels and related frame_idx
        self.index_labels.append("Scene 1")
        self.index_frames.append(self.shot_frame_idx[frame_idx])

        self.index_labels.append("          Shot "+str(shotNum))
        self.index_frames.append(self.shot_frame_idx[frame_idx])
        frame_idx = frame_idx + 1

        for i in range(1, len(self.shot_frames_bgr)):
            print(similarity_matrix[i - 1, i])
            if similarity_matrix[i - 1, i] > SCENE_THRESHOLD:
                current_scene.append(self.shot_frames_bgr[i])
                shotNum = shotNum + 1

                # Save Index labels and related frame_idx
                self.index_labels.append("          Shot "+str(shotNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])
                frame_idx = frame_idx + 1
                print(f"frame_idx {frame_idx}")
            else:
                # scenes.append(current_scene)
                current_scene = [self.shot_frames_bgr[i]]
                sceneNum = sceneNum + 1
                shotNum = 1

                print(f"frame_idx {frame_idx}")
                # Save Index labels and related frame_idx
                self.index_labels.append("Scene "+str(sceneNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])

                self.index_labels.append("          Shot "+str(shotNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])
                frame_idx = frame_idx + 1

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        # convert the frame data to a QImage for display in the label widget
        qimage = QImage(
            frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # display the frame in the label widget
        self.label.setPixmap(pixmap)

    def play_video(self):

        if not self.start:

            # create a thread to read the video and emit signals with the frame data
            self.video_thread = VideoThread(
                self.video_file_path, self.width, self.height, self.num_frames, self.fps, self.start_frame_idx,
                self.index_frames, self.listView)
            self.video_thread.update_frame.connect(self.update_frame)

            # start the thread to read the video
            self.video_thread.start()
            self.start = True

            # audio, get parameter later
            audioFile_path = self.audio_file_path

            # create an audio thread
            self.audio_thread = AudioThread(
                audioFile_path, self.start_frame_idx)
            # self.audio_thread.update_frame.connect(self.update_frame)
            self.audio_thread.start()

        if self.start:
            if self.paused:

                self.audio_thread.resume()
                self.video_thread.resume()
                self.paused = False
                self.btnPlay.setText(
                    QCoreApplication.translate("Dialog", u"Play", None))

    def pause_video(self):
        # set paused to True
        self.video_thread.pause()
        self.audio_thread.pause()
        if not self.paused:
            self.paused = True
            self.btnPlay.setText(
                QCoreApplication.translate("Dialog", u"Resume", None))

    def stop_video(self):
        # call stop function in the thread class
        self.video_thread.stop()
        self.audio_thread.stop()

        # reset the all related var
        self.start = False
        self.paused = False
        self.start_frame_idx = 0

        # select the first item in the list
        index = self.model.index(0)
        self.listView.setCurrentIndex(index)

        # When stop click, make play button's label 'Play'
        self.btnPlay.setText(
            QCoreApplication.translate("Dialog", u"Play", None))

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(
            QCoreApplication.translate("Dialog", u"Dialog", None))
        self.btnPlay.setText(
            QCoreApplication.translate("Dialog", u"Play", None))
        self.btnPause.setText(
            QCoreApplication.translate("Dialog", u"Pause", None))
        self.btnStop.setText(
            QCoreApplication.translate("Dialog", u"Stop", None))

    # Define a custom slot to handle the click event
    @Slot('QModelIndex')
    def on_item_clicked(self, index):

        idx = self.listView.currentIndex().row()
        frame_idx = self.index_frames[idx]
        print("idx: ", idx)

        # Call move_to_frame which kills running video and audio threads, and start new threads with the given frame index.
        self.move_to_frame(frame_idx)

    def move_to_frame(self, frame_idx: int):

        # if video started, stop and kill all threads
        if(self.start):
            # call stop function in the thread class
            self.video_thread.stop()
            self.audio_thread.stop()

            # reset the all related var
            self.start = False
            self.paused = False
            self.start_frame_idx = 0

            # wait for the both two threads to finish or be killed
            while self.video_thread.isRunning() or self.audio_thread.isRunning():
                pass

            # check if the tvideo and audio hreads were killed or has finished
            # Caution: make sure threads are killed before re-startint the thread.
            if self.video_thread.isFinished() and self.audio_thread.isFinished():
                print("Thread ended properly")
            else:
                print("Thread was safely terminated")

        # pass frame_idx into VideoThread and AudioThread to start Video and Audio
        self.start_frame_idx = frame_idx
        self.play_video()

    def close(self):

        # wait for the both two threads to finish or be killed
        while self.video_thread.isRunning() or self.audio_thread.isRunning():
            self.video_thread.stop()
            self.audio_thread.stop()

        # check if the tvideo and audio hreads were killed or has finished
        # Caution: make sure threads are killed before re-startint the thread.
        if self.video_thread.isFinished() and self.audio_thread.isFinished():
            print("Threads ended properly")
        else:
            print("Threads are still running")


class MainQDialog(QDialog):
    def __init__(self, args):
        super().__init__()

        # Set up the UI
        self.ui = Ui_Dialog(self, args)

    def closeEvent(self, event):
        # Custom behavior when the "X" button is pressed
        print("X button pressed. Closing the QDialog.")
        self.ui.close()

        event.accept()  # Accept the event to close the QDialog


if __name__ == '__main__':

    # QtApp
    app = QApplication(sys.argv)

    # Create and show the MainQDialog
    dialog = MainQDialog(sys.argv)
    dialog.show()

    # Start
    app.exec()
