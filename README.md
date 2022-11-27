# Computer Programming Project
## Table of Contents


- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Structure](#Structure)

## Background
- Base environment:
  - Operating system: win10, anaconda3.5.1
  - Python: base environment is Python 3.7
  - GPU: NVIDIA RTX2080ti 
  - CUDA: 10.1
  - cuDNN: 7.6 Needs to be adapted to CUDA

- Framework:
  - pytorch-gpu
  - TensorFlow-gpu
  - When installing pytorch and tensorflow, pay particular attention to the version, and consider the compatibility between hardware and software.
  
## Install
  The users need to install the libraries which show as follows:
  - sys/cv2/numpy/scipy/math/time/PIL/os.path/matplotlib.pyplot
- About pyQt5:   
  - qdarkstyle
  - from PyQt5.QtGui import QImage, QPixmap
  - from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
  - from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
  
## Usage
  The users can only run the file named "**myview.py**" to run the project. We have integrated all the code in this file except for the model training.
  
## Structure
  - Three interpolation-based methods:(They have already been all combined in the "**myview.py**")
    - Nearest_neighbor.py
    - Bilinear.py
    - bicubic.py
  - Deep learning based method:(In "**myview.py**", we use the trained model to reconstruct images)
    - createdata.py —— create the dataset for training
    - trainsrresnet.py —— train the model
    - test.py —— test the model
    - eval.py —— evaluate the model
    - utils.py
    
    
