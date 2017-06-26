from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def increase_heat(heatmap, labels, threshold):
    #print("threshold:", threshold)
    for label in range(1, labels[1]+1):
        heat = np.copy(heatmap)
        lbl = np.copy(labels[0])
        lbl[lbl != label] = 0
        lbl[lbl != 0] = 1
        # mask the heat map to only this label's pixels
        heat[lbl != 1] = 0
        # if the max is > threshold increase the heat by threshold
        #print("label:", label, "max:", np.max(heat))
        if np.max(heat) > threshold:
            heat[heat == 1] = threshold
        else:
            heat = np.zeros_like(heatmap)
        heatmap = heatmap + heat
        plt.imshow(heatmap, cmap='hot')
    return heatmap

# Draw the bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, labels, ret_boxes = False):
    # Iterate through all detected cars
    bbox = None
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        # note labels returns a 2D array with the labeled image in [0] & # of labels in [1]
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # filter out unrealistic boxes based on aspect ratio
        if abs((bbox[1][1]-bbox[0][1])/(bbox[1][0]-bbox[0][0])) < 2.5:
            bboxes.append(bbox)
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    if ret_boxes:
        return img, bboxes
    else:
        return img

def add_overlay(src, dst, spot, h, w):
    """add a w x h resized version of src to dst at the top of the image - up to 3 spot choices 0-2 with a 10px border"""
    # scale the image to the expected 0 to 255
    if np.max(src) == 0:
        scale = 1
    else:
        scale = int(255./np.max(src))

    if len(src.shape) == 2:
        overlay = np.dstack((src.astype(np.uint8), src.astype(np.uint8), src.astype(np.uint8)))
    else:
        overlay = src.astype(np.uint8)
    hot = cv2.applyColorMap(overlay*scale, cv2.COLORMAP_HOT)
    dst[10:h-10,spot*w+10:(spot+1)*w-10] = cv2.resize(hot, (w-20,h-20), interpolation = cv2.INTER_AREA)
