import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab



def text_histogram(image, horizontal=True):
    h, w = image.shape
    if horizontal:
        projection = np.sum(image,axis=1,keepdims=True)/255
    else:
        projection = (np.sum(image,axis=0,keepdims=True)/255).T

    return projection

def text_lines_segmentation(thresholded, th, tv):
    h, w = thresholded.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    h_projection = text_histogram(thresholded)
    plt.plot(range(h), h_projection)
    plt.title('h_projection')
    pylab.savefig('./output/18_h_projection', bbox_inches='tight')
    plt.clf()

    v_projection = text_histogram(thresholded, False)
    plt.plot(range(w), v_projection)
    plt.title('v_projection')
    pylab.savefig('./output/19_v_projection', bbox_inches='tight')
    plt.clf()

    first_y = 0
    for y in range(0, w):
        if v_projection[y] >= tv:
            first_y = y
            break

    last_y = w
    for y in range(w-1, -1, -1):
        if v_projection[y] >= tv:
            last_y = y
            break

    for x in range(0, h):
        for y in range(0, w):
            if h_projection[x] >= th and y > first_y and y < last_y:
                mask[x, y] = 255

    return mask

def text_characters_segmentation(thresholded, lines_mask, tv):
    boxes = []
    nb_components, output, _, _ = cv2.connectedComponentsWithStats(lines_mask, connectivity=4)

    for i in range(1, nb_components): # skip background
        component = ((output == i) * 255).astype(np.uint8)
        h, w = component.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        masked_lines_text = cv2.bitwise_and(thresholded, thresholded, mask=component)
        v_projection = text_histogram(masked_lines_text, False)

        for x in range(0, h):
            for y in range(0, w):
                if v_projection[y] >= tv:
                    mask[x, y] = 255

        characters_mask_single_lie = cv2.bitwise_and(mask, mask, mask=component)

        # extract bbox for each character from a single line of text
        nb_components2, _, stats2, _ = cv2.connectedComponentsWithStats(characters_mask_single_lie, connectivity=4)
        for j in range(1, nb_components2): # skip background
            x = stats2[j, cv2.CC_STAT_LEFT]
            y = stats2[j, cv2.CC_STAT_TOP]
            W = stats2[j, cv2.CC_STAT_WIDTH]
            H = stats2[j, cv2.CC_STAT_HEIGHT]
            boxes.append([x, y, W, H])

    return boxes


def draw_bbox_text_lines(image, mask):
    out = image.copy()
    nb_components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    # skip background
    for i in range(1, nb_components):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        W = stats[i, cv2.CC_STAT_WIDTH]
        H = stats[i, cv2.CC_STAT_HEIGHT]
        start_point = (x, y)
        end_point = (x + W, y + H)
        color = (0, 255, 0)
        cv2.rectangle(out, start_point, end_point, color, 2)
    return out

def draw_boxes(image, boxes):
    out = image.copy()
    for box in boxes:
        x, y, W, H = box
        start_point = (x, y)
        end_point = (x + W, y + H)
        color = (0, 255, 0)
        cv2.rectangle(out, start_point, end_point, color, 2)
    return out



def text_detection(binary_mask, original):

    # binary mask for the text! (background is black)
    binary_mask = (255 - binary_mask).astype(np.uint8)

    # segment into lines
    text_lines_mask = text_lines_segmentation(binary_mask, th=10, tv=5)

    # draw boxes around text lines
    cv2.imwrite('./output/20_text_lines_segmentation.jpg', draw_bbox_text_lines(original, text_lines_mask))

    # segment text characters
    text_characters_boxes = text_characters_segmentation(binary_mask, text_lines_mask, tv=3)

    # draw boxes around text lines
    cv2.imwrite('./output/21_text_characters_segmentation.jpg', draw_boxes(original, text_characters_boxes))

    return text_characters_boxes
