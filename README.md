# DA-VINCI-ENDOSCOPE-POSE-ESTIMATION
Estimation of endoscope camera motion trajectories from surgical videos with Deep Learning

The project aims to build a model that estimates the camera pose between 2 consecutive frames in order to predict the overall camera trajectory

<p align="center">
  <img alt="noise" src="https://github.com/AndreaNaclerio/DA-VINCI-ENDOSCOPE-POSE-ESTIMATION/assets/107640468/fc8a57ae-eae8-4046-bf5b-58d2e8ec8f14">
</p>

## DATASET
The dataset contains 27 endoscopic videos of porcine cadaver anatomy, captured with a da Vinci Xi endoscope and projector for depth mapping. Ground truth values in millimeters are provided, along with camera poses represented by 4x4 matrices. Images are resized to 512x640 pixels and normalized for computational efficiency.

<p align="center">
  <img alt="noise" src="https://github.com/AndreaNaclerio/DA-VINCI-ENDOSCOPE-POSE-ESTIMATION/assets/107640468/ae024879-beef-498a-abd5-53b85d646594">
</p>

## PRE-PROCESSING

## MODELS
To estimate the camera trajectory throughout the video frames, a model was trained to predict the final pose relative to the initial pose using consecutive frames. This simpler task allowed for reconstructing the overall camera path. The model architecture, Two Tails, consisted of two identical branches processing 2 different inputs (frames) in parallel. Each branch used convolutional, activation, and pooling layers to extract relevant features from RGB images.
In particular 2 different Deep Learning models have been tested:
- from scratch main block: CONV2D - BATCH NORMALIZATION - SQUEEZE AND EXPANTION BLOCK - ReLu - MAX POOLING)
- ResNet50
<p align="center">
  <img alt="noise" src="https://github.com/AndreaNaclerio/DA-VINCI-ENDOSCOPE-POSE-ESTIMATION/assets/107640468/5091f122-c0bf-473c-ab80-e1a1e618200a">
</p>


## RESULTS
The scratch model provides more accurate predictions overall, with only a slight offset in the y-axis and successful capture of the circular camera movement, though it struggles with fast and subtle movements. In contrast, the transfer-learning model captures the general direction but fails to predict accurately, especially in the Z-axis. Despite these challenges, both models successfully identify the overall trajectory shape.

<p float="left" align="center">
  <img src="https://github.com/AndreaNaclerio/DA-VINCI-ENDOSCOPE-POSE-ESTIMATION/assets/107640468/2cc3c44c-0fcd-48fe-9fdd-16cf62a0c3cf)" hspace="30"  width="500" heigth="500"/ >
  <img src="https://github.com/AndreaNaclerio/DA-VINCI-ENDOSCOPE-POSE-ESTIMATION/assets/107640468/4c401482-b4fc-498f-bead-4de8edbd0a5b" hspace="30"  width="500" heigth="500"/> 
</p>
