import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern



def calculate_hog(image):
    # 9, 8x8, 2x2
    fd, _ = hog(image, orientations=9, pixels_per_cell=(6,6),
                        cells_per_block=(6,6), block_norm='L2-Hys', visualize=True, transform_sqrt=True)
    hog_features = fd
    return hog_features.reshape((1, -1))

def calculate_lbp(image, hist=True):
    eps=1e-7
    radius = 1
    n_points = 8 * radius
    method = 'uniform'

    lbp = local_binary_pattern(image, n_points, radius, method)

    if hist:
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

    return np.array(lbp)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def write_and_print(file, name):
    cv2.imwrite(name + '.jpg', file)
    print(name)


def false_colors(image, nb_components):
    # Create false color image
    colors = np.random.randint(0, 255, size=(nb_components , 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    #colors[0] = [207, 59, 0]
    return colors[image]

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


def resize_with_max_ratio(image, max_h, max_w):
    if len(image.shape) > 2:
        w, h, ch = image.shape
    else:
        w, h = image.shape

    if (h > max_h) or (w > max_w):
        rate = max_h / h
        rate_w = w * rate
        if rate_w > max_h:
            rate = max_h / w
        image = cv2.resize(
            image, (int(h * rate), int(w * rate)), interpolation=cv2.INTER_CUBIC
        )
    return image
