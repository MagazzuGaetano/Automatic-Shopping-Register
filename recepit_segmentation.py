import cv2
import numpy as np
from scipy.fftpack import dct, idct
from utils import resize_with_max_ratio


def fillhole(input_image):
	'''
	input gray binary image  get the filled image by floodfill method
	Note: only holes surrounded in the connected regions will be filled.
	:param input_image:
	:return:
	'''
	im_flood_fill = input_image.copy()
	h, w = input_image.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	im_flood_fill = im_flood_fill.astype("uint8")
	cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
	im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
	img_out = input_image | im_flood_fill_inv
	return img_out 

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


def receipt_kmeans_segmentation(image):
    # Resize Image to speed up preprocessing
    image = resize_with_max_ratio(image, 1024, 1024)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Remove Noise
    denoised = cv2.GaussianBlur(image, (21, 21), 0)

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

    # Labeling connected components
    lcc = cv2.connectedComponentsWithStats(binary_mask, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = lcc

    # Keep only the component with the largest area
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    binary_mask = (labels == largest_label).astype("uint8") * 255

    # add padding
    pad = 32
    binary_mask = cv2.copyMakeBorder(binary_mask,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)

    # Morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=21)

    binary_mask = fillhole(binary_mask)

    binary_mask = cv2.erode(binary_mask, kernel, iterations=21)

    # remove padding
    h, w = binary_mask.shape
    binary_mask = binary_mask[pad:h-pad,pad:w-pad]

    # RGB -> GRAY
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    recepit = cv2.bitwise_and(image, image, mask=binary_mask) # masking
    return recepit, binary_mask

def recepit_threshold_segmentation(image):
    # Resize Image to speed up preprocessing
    image = resize_with_max_ratio(image, 1024, 1024)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # High pass filter
    frequencies = dct(dct(gray, axis=0), axis=1)
    frequencies[:2,:2] = 0
    gray = idct(idct(frequencies, axis=1), axis=0)
    gray = (gray - gray.min()) / (gray.max() - gray.min()) # renormalize to range [0:1]
    gray = np.asarray(gray * 255, dtype=np.uint8)
    cv2.imwrite('./output/1_HPF.jpg', gray)

    # Otsu Thresholding
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('./output/2_otsu.jpg', thresh1)

    # Labeling connected components
    (_, labels, stats, _) = cv2.connectedComponentsWithStats(thresh1, 4, cv2.CV_32S)

    # Keep only the component with the largest area
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    componentMask = ((labels == largest_label) * 255).astype("uint8")
    cv2.imwrite('./output/3_max_component.jpg', componentMask)

    # add padding
    pad = 32
    padded_mask = cv2.copyMakeBorder(componentMask,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)

    # Morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilate_mask = cv2.dilate(padded_mask, kernel, iterations=21)
    cv2.imwrite('./output/4_dilate.jpg', dilate_mask)

    fillhole_mask = fillhole(dilate_mask)
    cv2.imwrite('./output/5_fillhole.jpg', fillhole_mask)

    erode_mask = cv2.erode(fillhole_mask, kernel, iterations=21)
    cv2.imwrite('./output/6_erode.jpg', erode_mask)

    # remove padding
    h, w = erode_mask.shape
    erode_mask = erode_mask[pad:h-pad,pad:w-pad]

    recepit = cv2.bitwise_and(gray, gray, mask=erode_mask) # masking
    return recepit, erode_mask



# import os
# folder = './images'
# for filename in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder, filename))
#     recepit, mask = recepit_threshold_segmentation(img)
#     cv2.imwrite('./segmentation/{}'.format(filename), recepit)


