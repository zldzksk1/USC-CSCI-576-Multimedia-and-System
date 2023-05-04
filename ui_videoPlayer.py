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

import argparse

from scipy.signal import stft

import numpy as np
import cv2
import time

import wave

import sys

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QStringListModel, QThread, Signal, Slot)
from PySide6.QtGui import (QImage, QPixmap, QImage, QPixmap)
from PySide6.QtWidgets import (QApplication, QDialog, QListView,
                               QPushButton, QWidget, QLabel, QVBoxLayout)

from sklearn.metrics.pairwise import cosine_similarity


from threads.utils import Utils
from threads.videoThread import VideoThread
from threads.audioThread import AudioThread

########################
# Pre-defined Parameters
########################
# For SHOTS
SHOT_THRESHOLD = 15  # MAD #Can adjust it
SAME_SHOT_WINDOWS = 15  # Number of Indexes we assume the same shots #Can adjust it

# For SCENES
# Similarity #The larger, The smaller threshold(One Scene has more shots)
SCENE_COEFF_COLOR_HIST = 0.5
# Similarity #The larger, The smaller threshold(One Scene has more shots)
SCENE_COEFF_HOG = 0.81

# For SUBSHOTS
DETECT_SUBSHOT_WINDOWS = 10  # Unit: Seconds
SPECTRA_DIFFERENCE_THRESHOLD = 0.8  # 50% change in spectral energy


# Video Widget
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


# UI_Dialog
class Ui_Dialog(object):

    def __init__(self, Dialog, args):
        print("Parameters: ", args)

        # define the path to the RGB video file and the video parameters
        self.video_file_path = Path(args.video_file_path)
        self.audio_file_path = Path(args.audio_file_path)
        self.width, self.height = 480, 270
        self.fps = 30
        self.num_frames = (int)(self.calNumOfFrame())
        print(f"num_frames : {self.num_frames}")

        self.shot_frame_idx = []
        self.shot_frames_gray = []  # Shot's start frames(gray)
        self.shot_frames_bgr = []  # Shot's start frames(bgr)
        self.bgr_frames = []  # All frames(bgr)
        self.shot_frames_mean_bgr = []  # Shot's mean values of frames(bgr)
        self.shot_frames_med_bgr = []  # Shot's median frame(bgr)

        # For String view list
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
        print("**************************")
        print("***Start searching Shots.")
        print("**************************")

        shot_number = 0
        # open the video file for reading
        with open(self.video_file_path, "rb") as file:
            # read each frame of the video and display it
            for i in range(self.num_frames):

                # read the raw pixel data for the current frame
                raw_data = file.read(self.width * self.height * 3)

                # convert the raw data to a numpy array of pixel values
                pixels = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))  # pixels is BGR format

                # convert the RGB scale to Grey scale
                bgr_image = pixels
                gray_image = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                self.bgr_frames.append(pixels)

                # calculate the pixel difference between two frames
                if i > 0:

                    # Calculate mean of absolute difference
                    mad = Utils.meanAbsDiff(gray_image, prev_gray_image)

                    if mad > SHOT_THRESHOLD:
                        # Print MAD value
                        print("shot_number : ", shot_number+1,
                              ", MAD: ", mad,
                              ", Frame Index: ", i)
                        if (i - self.shot_frame_idx[shot_number-1] <= SAME_SHOT_WINDOWS):
                            # print("We have the same shots between same_shot_windows. Change past shot to new shot!!")
                            # self.shot_frame_idx[shot_number-1] = i
                            print(
                                "We have the same shots between same_shot_windows. Skip it!!")
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
        print("**************************")
        print(
            "***Start extrating Scenes and making subshots detecting abrupt audio changes.")
        print("**************************")

        # Calculate the mean of shots
        start_idx = 0
        end_idx = 0
        loop_count = len(self.shot_frame_idx)
        shot_frames = []

        for i in range(loop_count):
            start_idx = self.shot_frame_idx[i]

            if i < loop_count - 1:  # If not the last shot
                end_idx = self.shot_frame_idx[i + 1]-1
            else:  # Last shot
                end_idx = self.num_frames

            med_idx = (start_idx+end_idx) // 2

            self.shot_frames_med_bgr.append(self.bgr_frames[med_idx])

        # Define candidate frames as median shots
        candidate_color_frames_bgr = self.shot_frames_med_bgr
        candidate_hog_frames_bgr = self.shot_frames_med_bgr

        color_features = np.vstack([np.hstack((Utils.extract_color_histogram_features(
            frame))) for frame in candidate_color_frames_bgr])
        hog_features = np.vstack([np.hstack((Utils.extract_hog_features(
            frame))) for frame in candidate_hog_frames_bgr])

        # Calculate similarity matrix
        similarity_matrix_color = cosine_similarity(color_features)
        similarity_matrix_hog = cosine_similarity(hog_features)

        # Compute the mean and standard deviation of the distances for each feature
        similarity_mean_color = [np.mean(shot)
                                 for shot in similarity_matrix_color]
        similarity_std_color = [np.std(shot)
                                for shot in similarity_matrix_color]
        similarity_mean_hog = [np.mean(shot)
                               for shot in similarity_matrix_hog]
        similarity_std_hog = [np.std(shot) for shot in similarity_matrix_hog]

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

        self.detect_abrupt_sound_change(
            self.audio_file_path, self.shot_frame_idx[frame_idx - 1], self.shot_frame_idx[frame_idx], 30)

        for i in range(1, len(self.shot_frames_med_bgr)):
            if similarity_matrix_color[i - 1, i] > (similarity_mean_color[i] - SCENE_COEFF_COLOR_HIST * similarity_std_color[i]) \
                    and similarity_matrix_hog[i - 1, i] > (similarity_mean_hog[i] - SCENE_COEFF_HOG * similarity_std_hog[i]):
                current_scene.append(self.shot_frames_bgr[i])
                shotNum = shotNum + 1

            else:
                if similarity_matrix_color[i - 1, i] <= (similarity_mean_color[i] - SCENE_COEFF_COLOR_HIST * similarity_std_color[i]):
                    print("***The COLOR Features Difference is way more than before")
                if similarity_matrix_hog[i - 1, i] <= (similarity_mean_hog[i] - SCENE_COEFF_HOG * similarity_std_hog[i]):
                    print("***The HOG Features Difference is way more than before")

                current_scene = [self.shot_frames_bgr[i]]
                sceneNum = sceneNum + 1
                shotNum = 1

                # Save Index labels and related frame_idx
                print("***Scene Changed!!")
                self.index_labels.append("Scene "+str(sceneNum))
                self.index_frames.append(self.shot_frame_idx[frame_idx])

            # Save Index labels and related frame_idx
            self.index_labels.append("          Shot "+str(shotNum))
            self.index_frames.append(self.shot_frame_idx[frame_idx])
            frame_idx = frame_idx + 1

            if(len(self.shot_frame_idx) > frame_idx):
                self.detect_abrupt_sound_change(
                    self.audio_file_path, self.shot_frame_idx[frame_idx - 1], self.shot_frame_idx[frame_idx], 30)

            print(
                f"Index {sceneNum}-{shotNum}:({self.shot_frame_idx[i]}) {similarity_matrix_color[i - 1, i]} / {similarity_mean_color[i] - SCENE_COEFF_COLOR_HIST * similarity_std_color[i]}, {similarity_matrix_hog[i - 1, i]} / {similarity_mean_hog[i] - SCENE_COEFF_HOG * similarity_std_hog[i]}")

    def detect_abrupt_sound_change(self, audioPath, start_video_frame_idx, end_video_frame_idx, frame_rate):

        wave_file = wave.open(str(audioPath), 'rb')

        audio_frame_rate = wave_file.getframerate()

        # Define the start and end frame indices and window size for detecting sudden volume changes
        start_frame_idx = start_video_frame_idx
        end_frame_idx = end_video_frame_idx
        start_audio_idx = int(start_frame_idx * audio_frame_rate / frame_rate)
        end_audio_idx = int(end_frame_idx * audio_frame_rate / frame_rate)

        duration = (end_audio_idx - start_audio_idx) / audio_frame_rate
        if duration <= DETECT_SUBSHOT_WINDOWS:
            return
        # window size of 1 second
        window_size = int(audio_frame_rate * duration)
        threshold = SPECTRA_DIFFERENCE_THRESHOLD  # 50% change in spectral energy

        # Compute the STFT parameters
        fft_size = 1024
        hop_size = int(window_size / 2)

        # Loop through the audio in the specified range and detect sudden changes in the spectral characteristics
        num = 1
        for i in range(start_audio_idx, end_audio_idx, hop_size):
            data = wave_file.readframes(window_size)
            data_np = np.frombuffer(data, dtype=np.int16)
            if len(data_np) < window_size:
                break
            stft = np.abs(np.fft.rfft(data_np, fft_size))
            if i > start_audio_idx:
                prev_data = wave_file.readframes(window_size)
                prev_data_np = np.frombuffer(prev_data, dtype=np.int16)
                if len(prev_data_np) < window_size:
                    break
                prev_stft = np.abs(np.fft.rfft(prev_data_np, fft_size))
                spectral_diff = np.sum(
                    np.abs(stft - prev_stft)) / np.sum(prev_stft)
                if 1 > spectral_diff > threshold:
                    if num == 1:
                        video_idx = int(start_audio_idx *
                                        frame_rate / audio_frame_rate)
                        self.index_labels.append(
                            "                 Subshot " + str(num))
                        self.index_frames.append(video_idx)
                        num += 1
                    video_idx = int(i * frame_rate / audio_frame_rate)
                    self.index_labels.append(
                        "                 Subshot " + str(num))
                    self.index_frames.append(video_idx)
                    num += 1
            wave_file.setpos(i)

        wave_file.close()

    @ Slot(np.ndarray)
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
    @ Slot('QModelIndex')
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
            time.sleep(0.5)

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

        if self.start:
            self.video_thread.stop()
            self.audio_thread.stop()

        time.sleep(0.5)
        # check if the tvideo and audio hreads were killed or has finished
        if self.video_thread.isFinished() and self.audio_thread.isFinished():
            print("Threads ended properly")
        else:
            self.video_thread.start()
            self.video_thread.stop()
            self.video_thread.start()
            self.audio_thread.stop()
            if self.video_thread.isFinished() and self.audio_thread.isFinished():
                print("Threads ended properly")
            else:
                print("Threads are still running")


class MainQDialog(QDialog):
    def __init__(self, args):
        super().__init__()

        # Set up the UI
        self.ui = Ui_Dialog(self, args)

        self.setWindowTitle("Video Player")

    def closeEvent(self, event):
        # Custom behavior when the "X" button is pressed
        print("X button pressed. Closing the QDialog.")
        self.ui.close()

        event.accept()  # Accept the event to close the QDialog


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # For SHOTS
    SHOT_THRESHOLD = 15  # MAD #Can adjust it
    SAME_SHOT_WINDOWS = 15  # Number of Indexes we assume the same shots #Can adjust it

    # For SCENES
    # Similarity #The larger, The smaller threshold(One Scene has more shots)
    SCENE_COEFF_COLOR_HIST = 0.5
    # Similarity #The larger, The smaller threshold(One Scene has more shots)
    SCENE_COEFF_HOG = 0.81

    # For SUBSHOTS
    DETECT_SUBSHOT_WINDOWS = 10  # Unit: Seconds
    SPECTRA_DIFFERENCE_THRESHOLD = 0.8  # 50% change in spectral energy

    # For Input files. These are positional arguments, so it should come first before other arguments.
    parser.add_argument('video_file_path')
    parser.add_argument('audio_file_path')

    # For SCENES
    parser.add_argument('-c', '--coef-color', default=0.5,
                        type=float, dest='SCENE_COEFF_COLOR_HIST', help='Color Similarity. The larger, The smaller threshold(One Scene has more shots)')
    parser.add_argument('-H', '--coef-hog', default=0.81,
                        type=float, dest='SCENE_COEFF_HOG', help='HOG Similarity. The larger, The smaller threshold(One Scene has more shots)')

    # For SHOTS
    parser.add_argument('-s', '--shot', default=15,
                        type=float, dest='SHOT_THRESHOLD')
    parser.add_argument('-w', '--window-size', default=15,
                        type=int, dest='SAME_SHOT_WINDOWS')

    # For SUBSHOTS
    parser.add_argument('-sw', '--sub-window-size', default=10,
                        type=int, dest='DETECT_SUBSHOT_WINDOWS')
    parser.add_argument('-sd', '--spectra-diff', default=0.8,
                        type=float, dest='SPECTRA_DIFFERENCE_THRESHOLD')

    args = parser.parse_args()

    # QtApp
    app = QApplication()

    # Create and show the MainQDialog
    dialog = MainQDialog(args)
    dialog.show()

    # Start
    app.exec()
