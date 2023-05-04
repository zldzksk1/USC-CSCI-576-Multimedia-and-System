import numpy as np
import cv2

from skimage.feature import hog

# Utilities
class Utils():
    # Function to extract color histogram features from an image (HSV version)
    def extract_color_histogram_features(img, bins=8):

        # Convert the frame from BGR to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calculate a 3D color histogram with 8 bins per channel
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins],
                            [0, 180, 0, 256, 0, 256])

        cv2.normalize(hist, hist)

        return hist.flatten()

    # Function to extract HOG features from an image
    def extract_hog_features(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):

        # Convert the frame from BGR to GRAY color space
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the HOG feature (textture)
        hog_features = hog(gray_img, orientations=9,
                           pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

        return hog_features

    # Function to Calculate Absolute Difference
    def meanAbsDiff(img1: np.ndarray, img2: np.ndarray) -> float:
        # Calculate absolute difference between two continuous frames(Frame by Frame)
        abs_diff = cv2.absdiff(img1, img2)

        # Calculate mean of absolute difference
        mad = np.mean(abs_diff)
        return mad
