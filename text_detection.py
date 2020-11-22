import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import fillhole
import pylab


def h_and_v_hist(image, th, tv):
    h, w = image.shape

    v_projection = (np.sum(image,axis=0,keepdims=True)/255).T
    h_projection = np.sum(image,axis=1,keepdims=True)/255

    plt.plot(range(h), h_projection)
    plt.title('h_projection')
    pylab.savefig('11_h_projection', bbox_inches='tight')
    plt.clf()

    plt.plot(range(w), v_projection)
    plt.title('v_projection')
    pylab.savefig('12_v_projection', bbox_inches='tight')
    plt.clf()

    h_projection = np.uint8(h_projection > th).tolist()
    v_projection = np.uint8(v_projection > tv).tolist()

    return h_projection, v_projection

def histogram_segmentation(thresholded, th, tv):
    
    h_projection, v_projection = h_and_v_hist(thresholded, th, tv)
    
    h, w = thresholded.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for x in range(0, h):
        for y in range(0, w):
            if h_projection[x][0] == 1 and v_projection[y][0] == 1:
                mask[x, y] = 255

    return mask


# input:
# - la maschera di ogni singolo carattere in una riga di testo
# - l'immagine RGB per poter croppare successivamente i caratteri
# - le soglie per la segmentazione del testo th e tv
# output: 
# - una lista di immagini (i singoli caratteri croppati)
def characters_detection(characters_mask, rgb_image, bbx):
    res_characters = []

    # Labeling Componenti Connesse
    num_labels2, labels_im2 = cv2.connectedComponents(characters_mask)

    # per ogni componente righa croppo i singoli caratteri
    # range(1, labels2) saltare la prima componente chè sarebbe il background
    for label2 in range(1, num_labels2):

        # maschera singolo carattere
        char_mask = np.uint8(labels_im2 == label2)
        _,char_mask = cv2.threshold(char_mask,0,255,cv2.THRESH_BINARY)

        # croppare il rettangolo della maschera singolo carattere
        x1, y1, w1, h1 = tuple(cv2.boundingRect(char_mask))
        char_mask_cropped = char_mask[y1:y1+h1, x1:x1+w1]

        # croppare il rettangolo sull'immagine rgb
        (x, y, w, h) = bbx
        rgb_cropped_line = rgb_image[y:y+h, x:x+w]
        rgb_cropped = rgb_cropped_line[y1:y1+h1, x1:x1+w1]

        res_characters.append(rgb_cropped)

    return res_characters

# input
# - un immagine binaria e una RGB
# - due soglie th e tv per la segmentazione del testo
# output: 
# - una lista con i singoli caratteri croppati da ogni riga di testo 
def text_detection(thresholded, rgb_image, th, tv):
    res_characters = []

    # trovo la maschera delle righe di testo orizzontali
    text_mask = histogram_segmentation(thresholded, th, tv)

    # dilate orizzontale
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    text_mask = cv2.dilate(text_mask, kernel, iterations=3)

    # Labeling Componenti Connesse
    num_labels, labels_im = cv2.connectedComponents(text_mask)

    # Scorro le righe di testo trovate nell'immagine
    # range(2, n-1) per evitare la prima componente e l'ultima
    for label in range(2, num_labels-1):

        # ottengo la maschera per la i-esima riga di testo    
        row_mask = np.uint8(labels_im == label)
        _,row_mask = cv2.threshold(row_mask,0,255,cv2.THRESH_BINARY)

        # ottengo il testo nella riga selezionata
        row_text_mask = thresholded & row_mask

        # croppo bounding box
        x, y, w, h = tuple(cv2.boundingRect(row_text_mask))
        row_text_mask = row_text_mask[y:y+h, x:x+w]

        # applico una dilate per evitare che ci siano linee verticali senza pixel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)) 
        dilated = cv2.dilate(row_text_mask, kernel, iterations = 1)

        # ottengo la maschera per tutti i caratteri nella riga di testo corrente
        characters_mask = histogram_segmentation(dilated, th=0, tv=0)

        # ottengo i caratteri croppati nella riga di testo corrente
        characters = characters_detection(characters_mask, rgb_image, bbx=(x,y,w,h))

        res_characters = res_characters + characters

    return res_characters