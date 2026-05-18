import sys
print('python', sys.version)

import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device', torch.cuda.get_device_name(0))

import cv2
print('cv2', cv2.__version__)

import ultralytics
print('ultralytics', ultralytics.__version__)

import onnxruntime as ort
print('onnxruntime', ort.__version__)

import bettercam, supervision, streamlit
print('bettercam/supervision/streamlit import ok')
