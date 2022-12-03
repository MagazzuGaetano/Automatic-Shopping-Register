import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
from recepit_segmentation import recepit_threshold_segmentation
from text_detection import text_detection
from utils import write_and_print
from skimage.morphology import convex_hull_image
from skimage.segmentation import clear_border
from skimage import filters
import pytesseract


#####################################################################################################################################################

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

def remove_recepit_border(skewed, thresholded):
    # Gray Image (Recepit Skewed)
    receipt_mask_skewed = cv2.cvtColor(skewed, cv2.COLOR_RGB2GRAY)
    # Binarization (Recepit Skewed Mask)
    receipt_mask_skewed[np.where(receipt_mask_skewed != 0)] = 255
    # Gradient take the border
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    gradient = cv2.morphologyEx(receipt_mask_skewed, cv2.MORPH_GRADIENT, kernel)
    # Compliment of (Recepit Skewed Mask)
    not_recepit_mask = np.uint8(np.invert(receipt_mask_skewed))
    # Sum Gradient and Recepit Skewed Mask then dilate them
    not_border_mask = cv2.dilate((gradient + not_recepit_mask), kernel, iterations=10)
    # Border Mask
    border_mask = np.uint8(np.invert(not_border_mask))
    # Finally Masking
    return thresholded & border_mask

#####################################################################################################################################################



#####################################################################################################################################################

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def hough_transform(image, rho=1, theta=np.pi / 180, threshold=30):
    '''
    parameters:
    @rho: Distance resolution of the accumulator in pixels.
    @theta: Angle resolution of the accumulator in radians.
    @threshold: Only lines that are greater than threshold will be returned.
    '''
    return cv2.HoughLines(image, rho=rho, theta=theta, threshold=threshold)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    line_length=100000 #impostare uguale alla diagonale
    for line in lines:
        for rho, theta in line:
            t = theta * 180 / np.pi
            t = np.mod(t, 90)
            k = 30

            if t >= 45 - k and t <= 45 + k:
                continue

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + line_length * (-b))
            y1 = int(y0 + line_length * (a))
            x2 = int(x0 - line_length * (-b))
            y2 = int(y0 - line_length * (a))
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def corner_detection(image, T=0.01, min_dist=50):
    corners_points = []
    corners_image = np.zeros_like(image)

    # detect corners
    pts_copy = cv2.goodFeaturesToTrack(image, 4, T, min_dist)
    pts_copy = np.int0([pt[0] for pt in pts_copy])

    # compute the distance from each corner to every other corner
    euclidian_dist = lambda pt1, pt2 : np.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)

    # if the points are not 4 we return zero points!
    if len(pts_copy) == 4:

        # sort coordinates_tuples (tl, tr, bl, br)
        [h, w] = image.shape
        tl_index = np.asarray([euclidian_dist(pt, (0, 0)) for pt in pts_copy]).argmin()
        tr_index = np.asarray([euclidian_dist(pt, (w, 0)) for pt in pts_copy]).argmin()
        bl_index = np.asarray([euclidian_dist(pt, (0, h)) for pt in pts_copy]).argmin()
        br_index = np.asarray([euclidian_dist(pt, (w, h)) for pt in pts_copy]).argmin()
        corners_points = np.asarray([pts_copy[tl_index], pts_copy[tr_index], pts_copy[br_index], pts_copy[bl_index]])

        for pt in corners_points:
            cv2.circle(corners_image, tuple(pt), 10, 255, -1)

    return corners_points, corners_image

def warp_image(img, src):
    # invert points coordinates
    src = np.array([[point[0], point[1]] for point in src], dtype="float32")
    (tl, tr, br, bl) = src

    w1 = int(np.hypot(bl[0] - br[0], bl[1] - br[1]))
    w2 = int(np.hypot(tl[0] - tr[0], tl[1] - tr[1]))

    h1 = int(np.hypot(tl[0] - bl[0], tl[1] - bl[1]))
    h2 = int(np.hypot(tr[0] - br[0], tr[1] - br[1]))

    max_w = np.max([w1, w2])
    max_h = np.max([h1, h2])

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (max_w, max_h))

    return warp


def correct_distorsion(recepit, recepit_mask):
    # add padding
    pad = 32
    recepit_mask = cv2.copyMakeBorder(recepit_mask,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)

    # convex hull
    convex_hull = (convex_hull_image(recepit_mask) * 255).astype(np.uint8)
    cv2.imwrite('./output/7_convexhull.jpg', convex_hull)

    # centroid of my mask edited with morphological operations and convexhull
    x,y,w,h = cv2.boundingRect(convex_hull)
    centroid = (x+round(w/2), y+round(h/2))

    # Canny edge detection
    (T, _) = cv2.threshold(convex_hull, 0, 255, cv2.THRESH_OTSU)
    canny = cv2.Canny(convex_hull, T * 0.5, T)
    cv2.imwrite('./output/8_canny.jpg', canny)

    # Hough Detection
    hough_lines = hough_transform(canny, threshold=100)
    hough_out = draw_lines(np.zeros(convex_hull.shape), hough_lines, color=255, thickness=3)
    cv2.imwrite('./output/9_hough.jpg', hough_out)

    # Invert hough mask
    invert_hough = 255 - hough_out
    invert_hough = ((invert_hough > 0) * 255).astype(np.uint8)
    cv2.imwrite('./output/10_invhough.jpg', invert_hough)

    # Get the closer component to the mask centroid
    nb_components, output, _, centroids = cv2.connectedComponentsWithStats(invert_hough, connectivity=4)
    dist = lambda p1, p2 : np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    min_label, min_dist = min([(i, dist(centroids[i], centroid)) for i in range(1, nb_components)], key=lambda x: x[1])
    min_dist_bw_label = ((output == min_label) * 255).astype(np.uint8)
    cv2.imwrite('./output/11_min_dist_bw_label.jpg', min_dist_bw_label)

    # Morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Dilate
    min_dist_bw_label = cv2.dilate(min_dist_bw_label, kernel, iterations=21)
    cv2.imwrite('./output/12_dilate.jpg', min_dist_bw_label)

    # Corner Detection
    _,_,w1,h1 = cv2.boundingRect(min_dist_bw_label)
    min_dist = min(w1, h1) * 0.75

    # corner detection
    corners_points, corners_image = corner_detection(min_dist_bw_label, T=0.01, min_dist=min_dist)

    # remove padding
    h, w = min_dist_bw_label.shape
    min_dist_bw_label = min_dist_bw_label[pad:h-pad,pad:w-pad]

    # Image Warping
    # if the corners detected are not 4 the warped image is not corrected!
    if len(corners_points) != 4:
        recepit = cv2.bitwise_and(recepit, recepit, mask=min_dist_bw_label) # masking
        un_warped = recepit
    else:
        recepit = cv2.bitwise_and(recepit, recepit, mask=min_dist_bw_label) # masking
        un_warped = warp_image(recepit, corners_points) # image distortion correction
        cv2.imwrite('./output/13_un_warped.jpg', un_warped)

    return un_warped

def ocr_preprocessing(recepit):
    # Sauvola
    mask = recepit < filters.threshold_sauvola(recepit, 31, 0.1)
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite('./output/14_sauvola.jpg', mask)

    # Clear border to remove possible background
    mask = clear_border(mask)
    cv2.imwrite('./output/15_clear_border.jpg', mask)

    # Sharpening
    blur = cv2.GaussianBlur(mask, (1, 1), 0)
    recepit = 255 - (255 - recepit) * blur
    cv2.imwrite('./output/16_sharpening.jpg', recepit)

    # Gamma Correction
    recepit = adjust_gamma(recepit, 1/10)
    cv2.imwrite('./output/17_gamma_correction.jpg', recepit)

    return recepit


def new_pipeline(image, method=0):

    # recepit segmentation
    recepit, recepit_mask = recepit_threshold_segmentation(image)

    un_warped = correct_distorsion(recepit, recepit_mask)

    # recepit preprocessing
    preprocess_recepit = ocr_preprocessing(un_warped)

    # OCR
    text = '...'
    if method == 0:
        text = pytesseract.image_to_string(preprocess_recepit, lang='ita')
    else:
        # Text Detection (List of single character's cropped)
        characters_boxes = text_detection(preprocess_recepit, preprocess_recepit)
        # predict characters with svm classifier + hog
        pass

    return text

#####################################################################################################################################################


# read the image
image = cv2.imread('./images/recepit.png')

# Extract Text
text = new_pipeline(image)

print(text)

