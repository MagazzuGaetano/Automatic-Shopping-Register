#################################################################################################
# LIMITAZIONI
# 1) SFONDO SENZA ALTRI OGGETTI
# 2) SFONDO ABBASTANZA SCURO
# 3) ROTAZIONE

# PROBLEMI
# 1) problemi di segmentazione dello scontrino (immagine 3 con la fillholes).
# 2) rimangono i bordi dello scontrino sia in verticale sia horizontalmente.
# 3) descriminare i ritagli dalle lettere da quelli senza (se ci sono tanti pixel "0" è sul bordo)
# 4) alcune scritte sono talmente vicine da risultare ritagli che non sono singole lettere
##################################################################################################

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
#import re
#from scipy import ndimage
#from skimage.filters import unsharp_mask
from recepit_segmentation import receipt_segmentation
from text_detection import text_detection
from utils import write_and_print


def illumination_equalization(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final

def find_score(image, angle):
    data = inter.rotate(image, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def correct_skewness(image, delta=1, limit=5):
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(image, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    res = inter.rotate(image, best_angle, reshape=False, order=0)
    #res = im.fromarray((255 * res).astype("uint8")).convert("RGB")

    return res



# read the image
image = cv2.imread('./Images/recepit.jpg')

# receipt segmentation
receipt, _ = receipt_segmentation(image)
write_and_print(receipt, '1_receipt segmentation')

# Skew Correction
skewed = correct_skewness(receipt)
write_and_print(skewed, '2_skewed')

# CLAHE illumination equalization
cl_img = illumination_equalization(skewed)
write_and_print(cl_img, '3_illumination_equalization')

# Gray Image
gray = cv2.cvtColor(cl_img, cv2.COLOR_RGB2GRAY)
write_and_print(gray, '4_gray')

# Adaptive Threshold
thresholded = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
write_and_print(thresholded, '5_threshold')

# Text Detection (Lista dei singoli caratteri croppati)
characters = text_detection(thresholded, skewed, th=30, tv=10) #th=40

x = 0
for char in characters:
    cv2.imwrite('./output/{}_out.png'.format(x), char)
    x = x + 1

# SVM + HOG Classification
