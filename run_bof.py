from src.bof import BOF
from src.databases import Databases
import cv2 as cv
import os

cv.ocl.setUseOpenCL(True)

os.environ['KEYPOINT_DETECTOR_TYPE'] = "DENSE"
os.environ['DENSE_DETECTOR_STEP'] = '7'
os.environ['VOCABULARY_SIZE'] = '2000'
os.environ['TRAINING_SIZE_PER_LABEL'] = '6000'

imagens, labels = Databases.get_minist('./')

bof = BOF(imagens, labels)

bof.run()
