import numpy as np
import time

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QStringListModel, QThread, Signal, Slot)
from PySide6.QtGui import (QImage, QPixmap, QImage, QPixmap)
from PySide6.QtWidgets import (QApplication, QDialog, QListView,
                               QPushButton, QWidget, QLabel, QVBoxLayout)

import cv2

#  Video Thread Class
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
        # Open the video files
        with open(self.file_path, "rb") as file:

            # Seek the start frame
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
                if curr_idx < len(self.index_frames) and curr_frame_number == self.index_frames[curr_idx]:
                    # Whatever it is scene->shot, shot->subshot, if they have same frame number, the last one should be selected.
                    while curr_idx < len(self.index_frames)-1 and curr_frame_number == self.index_frames[curr_idx+1]:
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
