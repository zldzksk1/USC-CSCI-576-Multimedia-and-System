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

# Load pre-trained VGG-16 model
#model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Extract features from an image
'''
def extract_features(img):
    img = cv2.resize(img, (224, 224))
    if len(img.shape) == 2 or img.shape[2] == 1:  # If the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features

# Function to extract color histogram features from an image
def extract_features(img, bins=8):
    if len(img.shape) == 2 or img.shape[2] == 1:  # If the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    hist = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
'''

# Function to extract color histogram features from an image
def extract_color_histogram_features(img, bins=8):
    if len(img.shape) == 2 or img.shape[2] == 1:  # If the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    hist = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to extract HOG features from an image
def extract_hog_features(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img=img
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
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

    def __init__(self, file_path: str, width: int, height: int, num_frames: int, fps: int, startIdx: int):
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

    def run(self):
        # open the video file for reading
        with open(self.file_path, "rb") as file:

            #Set the start frame
            file.seek(self.start_frame_idx * self.width * self.height * 3)

            # read each frame of the video and display it
            for i in range(self.start_frame_idx, self.num_frames):

                while self.threadPaused:
                    # print("timeSleep")
                    time.sleep(0.1)

                if self.stop_event:
                    break

                # read the raw pixel data for the current frame
                raw_data = file.read(self.width * self.height * 3)

                # convert the raw data to a numpy array of pixel values
                pixels = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                # convert the RGB pixels to BGR format for displaying in OpenCV
                bgr_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                # emit a signal with the current frame data
                self.update_frame.emit(bgr_image)

                # wait for the specified number of milliseconds before showing the next frame
                time.sleep(1 / self.fps)

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
        self.fps, self.num_frames = 30, 8682

        self.shot_threshold = 20
        self.same_shot_windows = 3
        self.shot_strings = []
        self.shot_frame_idx = []
        self.shot_frames = []

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

                # convert the RGB pixels to BGR format for displaying in OpenCV
                bgr_image = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

                # calculate the pixel difference between two frames
                if i > 0:

                    # Calculate mean of absolute difference
                    mad = meanAbsDiff(bgr_image, prev_bgr_image)

                    if mad > self.shot_threshold:
                        # Print MAD value
                        print("shot_number : ", shot_number+1,
                               ", MAD: ", mad,
                               ", Frame Index: ", i)
                        if shot_number > 1:   
                            if (i-self.shot_frame_idx[shot_number-1] <= self.same_shot_windows):
                                #print("We have the same shots between same_shot_windows. Change past shot to new shot!!")
                                #self.shot_frame_idx[shot_number-1] = i
                                print("We have the same shots between same_shot_windows. Skip it!!")
                            else :
                               # Add it to the array.
                                shot_number = shot_number + 1
                                self.shot_strings.append("Shot" + str(shot_number))
                                self.shot_frame_idx.append(i) 
                                self.shot_frames.append(bgr_image)
                                
                        else :
                            # Add it to the array.
                            shot_number = shot_number + 1
                            self.shot_strings.append("Shot" + str(shot_number))
                            self.shot_frame_idx.append(i)
                            self.shot_frames.append(bgr_image)

                else:
                    # For the first frame we do not calculate the pixel difference and just add it to the array.
                    shot_number = shot_number + 1
                    self.shot_strings.append("Shot" + str(shot_number))
                    self.shot_frame_idx.append(i)
                    self.shot_frames.append(bgr_image)

                # store the current frame as the previous frame for the next iteration
                prev_bgr_image = bgr_image.copy()

        ##########################
        # Extract Scene grouping shots
        ##########################
        # Extract features from frames
        #frame_features = np.vstack([extract_features(frame) for frame in self.shot_frames])
        frame_features = np.vstack([np.hstack((extract_color_histogram_features(frame), extract_hog_features(frame))) for frame in self.shot_frames])


        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(frame_features)

        # Group shots into scenes using a threshold
        sceneThreshold = 0.7  # Adjust this value based on your requirements        
        #scenes = []
        shotNum = 1
        sceneNum = 1
        frame_idx = 0
        current_scene = [self.shot_frames[0]]

        #Save Index labels and related frame_idx
        self.index_labels.append("Scene 1")
        self.index_frames.append(self.shot_frame_idx[frame_idx])

        self.index_labels.append("          Shot "+str(shotNum))
        self.index_frames.append(self.shot_frame_idx[frame_idx])
        frame_idx = frame_idx + 1

        for i in range(1, len(self.shot_frames)):
            print(similarity_matrix[i - 1, i])
            if similarity_matrix[i - 1, i] > sceneThreshold:
                current_scene.append(self.shot_frames[i])
                shotNum = shotNum + 1
                
                #Save Index labels and related frame_idx
                self.index_labels.append("          Shot "+str(shotNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])
                frame_idx = frame_idx + 1
                print(f"frame_idx {frame_idx}")
            else:
                #scenes.append(current_scene)
                current_scene = [self.shot_frames[i]]
                sceneNum = sceneNum + 1
                shotNum = 1

                print(f"frame_idx {frame_idx}")
                #Save Index labels and related frame_idx
                self.index_labels.append("Scene "+str(sceneNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])

                self.index_labels.append("          Shot "+str(shotNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])
                frame_idx = frame_idx + 1

        # Add the last scene
        #scenes.append(current_scene)

        # Add strings to the model
        self.model = QStringListModel()
        self.model.setStringList(self.index_labels)

        # Set the model on the list view
        self.listView.setModel(self.model)

        # close the OpenCV window
        cv2.destroyAllWindows()

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        # convert the frame data to a QImage for display in the label widget
        qimage = QImage(
            frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # display the frame in the label widget
        self.label.setPixmap(pixmap)

    def play_video(self):
        print("play_video")

        if not self.start:

            # create a thread to read the video and emit signals with the frame data
            self.video_thread = VideoThread(
                self.video_file_path, self.width, self.height, self.num_frames, self.fps, self.start_frame_idx)
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
        print("stop video!")
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

    # Define a custom slot to handle the click event
    @Slot('QModelIndex')
    def on_item_clicked(self, index):

        idx = self.listView.currentIndex().row()
        frame_idx = self.index_frames[idx-1]

        # Call move_to_frame which kills running video and audio threads, and start new threads with the given frame index.
        self.move_to_frame(frame_idx)

        # Below is just for debugging.
        # QMessageBox.information(
        #     self.listView, 'Item Clicked', f'You clicked on {item}')

    def move_to_frame(self, frame_idx: int):

        # if video started, stop and kill all threads
        if(self.start):
            #self.stop_video()
            print("if runing, make it stop video!")
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
        #self.ui.setupUi(self)

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
