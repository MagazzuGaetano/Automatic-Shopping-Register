import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import write_and_print, fillhole

def color_quantization(image, NCLUSTERS, NROUNDS):

    height, width, channels = image.shape
    samples = np.zeros([height*width, 3], dtype = np.float32)
    count = 0

    for x in range(height):
        for y in range(width):
            samples[count] = image[x][y]
            count += 1
    
    compactness, labels, centers = cv2.kmeans(samples,
                                        NCLUSTERS, 
                                        None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
                                        NROUNDS, 
                                        cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    image2 = res.reshape((image.shape))

    return image2, centers

def receipt_segmentation(image):

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Remove Noise
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 10) #15

    # RGB -> HSV
    hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)

    # kmeans color quantization
    output, labels = color_quantization(hsv, 2, 1)

    l1 = labels[0][2]
    l2 = labels[1][2]

    label = l1
    if l2 > l1:
        label = l2
    
    binary_mask = np.uint8(output[:,:,2] == label)
    
    '''plt.imshow(output)
    plt.title('kmeans color quantization HSV')
    plt.show()

    plt.imshow(binary_mask, cmap='gray')
    plt.title('HSV(value)')
    plt.show()

    plt.imshow(binary_mask, cmap='gray')
    plt.title('RESULT')
    plt.show()'''

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    #plt.imshow(binary_mask, cmap='gray')
    #plt.title('dilate')
    #plt.show()

    binary_mask = fillhole(binary_mask)

    #plt.imshow(binary_mask, cmap='gray')
    #plt.title('fillhole')
    #plt.show()
    
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    #plt.imshow(binary_mask, cmap='gray')
    #plt.title('erode')
    #plt.show()

    for i in range(0, 3):
        denoised[:,:,i] = denoised[:,:,i] & binary_mask

    return denoised, binary_mask
