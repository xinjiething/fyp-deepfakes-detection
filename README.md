# FaceSpot

## Introduction

Final Year Project for Bachelor of Computer Science, Monash University (S1/2020) with topic DeepFake Detection using Convolutional Neural Network

Team Members: Chan Jun Cheng, Thing Xin Jie, Yap Hao Zher

Project Supervisor: Prof. Ir. Dr. Raphael C.-W. Phan, CEng, FHEA (raphael.phan@monash.edu)


## Overview

A DeepFake detection program created by training the MesoNet architecture (https://github.com/dariusaf/mesonet) with negative samples generated which simulates the resolution inconsistencies (https://github.com/danmohaha/CVPRW2019_Face_Artifacts) created as a by-product of the current DeepFake algorithm.

## Requirements

- Ubuntu 16.04
- Python 2.7 and the following dependencies:
```
NumPy and SciPy
Pillow
Dlib 19.16.0
OpenCV 3.4.0
```
- CUDA 8.0
- cuDNN v6.0
- Tensorflow 1.4.0
- Keras 2.1.5


Additionally, if intending to launch the graphical user interface built using PyQt5, run these commands:

```
apt install -y python-dev

# dependencies for PyQt5
pip install enum34
add-apt-repository ppa:beineri/opt-qt-5.12.0-xenial
apt-get update
apt-get install -y build-essential libgl1-mesa-dev qt512-meta-minimal qt512webengine qt512svg

# install SIP
wget https://www.riverbankcomputing.com/static/Downloads/sip/4.19.14/sip-4.19.24.tar.gz
tar -xvzf sip-4.19.24.tar.gz
cd sip-4.19.14
python configure.py --sip-module=PyQt5.sip
make -j 4
make install

# install PyQt5
wget https://www.riverbankcomputing.com/static/Downloads/PyQt5/5.12/PyQt5-5.15.1.dev2008271829.tar.gz
tar -xvzf PyQt5-5.15.1.dev2008271829.tar.gz
cd PyQt5-5.15.1.dev2008271829
LD_LIBRARY_PATH=/opt/qt512/lib python configure.py --confirm-license --disable=QtNfc --qmake=/opt/qt512/bin/qmake QMAKE_LFLAGS_RPATH=
make -j 4
make install
```

Our project is done using Windows Subsytem for Linux. Main reason for this is it is easy to install and use (it's on Microsoft Store), and we can access our local machine's filesystem easily without needing to synchronise. However, a disadvantage for this is it lacks a graphical interface, which means that we need the help of external program in order to run the ui. To export display from linux terminal, we need to install XMing(Steps can be referred to this video: https://www.youtube.com/watch?v=3_iGSpyGswo,
Link to download XMing: https://drive.google.com/file/d/15vp-rQ79o_cSTwCGQbkl42cgpkq2zn1P/view).

Steps to start the UI
1) Launch Ubuntu 16.04 terminal.
2) Launch XMing.
3) Run the command "export DISPLAY=:0.0" to export display from linux terminal, here 0.0 indicates the XMing server and can be verified by hovering over the icon at the bottom right.
4) cd to the repository that was downloaded.
5) Run the command "python facespot.py" to start the UI.


## Preprocess, training and testing paths

By default, the folder structure recognized by the code as-is is as follows:

```
.
├── 8a_mesoinception4.h5 (model weights)
├── imgs (contains all images to be processed into pos- and neg- samples in its root)
|   ├── pos (generated pos- samples)
|   └── neg (generated neg- samples)
├── train_imgs (to be manually filled with pos- and neg- training data)
|   ├── positive
|   └── negative
├── test_imgs (to be manually filled with pos- and neg- training data)
|   ├── positive
|   └── negative
└── dlib_model
    └── shape_predictor_68_face_landmarks.dat
    
```
    
## Workflow

1. Generate positive and negative samples by giving preprocess.py a folder filled with images. All output images will be resized to 256x256px, and negative samples will be applied DeepFake simulation.

2. Manually split generated images into desired train-test ratio, and place respective data into the train_imgs and test_imgs folders, separated by pos- and neg-.

3. Run train.py and test.py.
