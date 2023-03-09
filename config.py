import cv2

#Arquivo
PATH = 'data/images/'
FILE = 'poster'
FORMAT = '.jpg'

#Língua
LANG = 'por'

#Processamento
NOISE_REDUCTION_STRENGTH = 2

GAUSSIAN_A = 10
GAUSSIAN_B = 10

#thresholding method
THRESHOLDING = cv2.THRESH_BINARY

#ERODE and DILATE sizes (pixels)
ERODE_X = 1
ERODE_Y = 1
DILATE_X = 1
DILATE_Y = 1