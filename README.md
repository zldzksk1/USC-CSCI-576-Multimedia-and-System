# CSCI-576---Video-Player
Final project for the CSCI 576 - Multimedia System

Our group project entails the implementation of a sophisticated video player with four primary features: play, pause, stop, and seamless scene navigation.<br>

# Shot and Scene Detection / Scene segmentation using Cosine Similarity
- Divide Shots comparing the differences between adjacent frames
- Grouping shots, and define scenes

Implementation:
Shot
- MAD(Mean Absolute Difference) 
Scene: 
- Color histogram: Comparing the distribution of colors in an image
- Histogram of Oriented Gradients (HOG): Capturing the distribution of gradient orientations (texture)in an image
- Cosine similarity: Calculating the cosine of the angle between two non-zero vectors

#Subshot Detection
- Emphasize the change of particular emotion or mood in a scene.

Imeplenmetation
- Short Time Fourier of the audio signal
- Compare the audio signal during the window size

Player images <br>
<div style="width: 300px;">
  <img src="https://github.com/zldzksk1/USC-CSCI-576-Multimedia-and-System/blob/main/videoPlayer2.png" alt="Player Image" style="max-width: 200px; height: auto;">
  <img src="https://github.com/zldzksk1/USC-CSCI-576-Multimedia-and-System/blob/main/videoPlayer.png" alt="Player Image" style="max-width: 200px; height: auto;">
<div>
