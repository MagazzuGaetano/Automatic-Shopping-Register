import cv2
import numpy as np

def write_and_print(file, name):
    cv2.imwrite(name + '.jpg', file)
    print(name)

def hue_labels_mapping(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def masking_rgb(image, mask):
    for i in range(0, 3):
        image[:,:,i] = image[:,:,i] & mask
    return image

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