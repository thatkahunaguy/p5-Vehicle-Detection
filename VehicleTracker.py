from pipeline import pipeline
from collections import deque
import numpy as np
import cv2

class VehicleTracker(object):
    """
    Track the lane in a series of consecutive frames.
    """

    def __init__(self, svc, X_scaler, orient, pix_per_cell,
                cell_per_block, spatial_size, hist_bins, threshold=2, cspace='BGR2YCrCb'):
        """
        Initialize a tracker object.

        """
        self.first_frame = True
        # track the last 20 frames of heatmaps including false positives
        # may eliminate this but useful for false positive reduction
        self.heatmap = deque(maxlen=20)
        # track the last 15 frames of thresholded heatmaps
        self.thresh_heatmap = deque(maxlen=15)
        # track the last 15 frames of bounding boxes
        self.bounding_boxes = deque(maxlen=15)
        # initialize the first frame number
        self.frame = 1
        # store the classifier, scaler, HOG, spatial, & color feature parameters
        self.svc = svc
        self.X_scaler = X_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.threshold = threshold
        self.cspace = cspace
        # save a frame label for the heatmap overlay labels
        self.frame_label = self.make_frame_label()

    def make_frame_label(self):
        frame_label = np.zeros((720,1280,3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # note had to add the line type and increase thickness to make the text readable
        cv2.putText(frame_label, "Single Frame", (100, 40), font, 1, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(frame_label, "Heatmap Deque Sum", (475, 40), font, 1, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(frame_label, "Thresholded Sum", (950, 40), font, 1, (255,255,255), 3, cv2.LINE_AA)
        return frame_label

    def get_heatmap_sum(self):
        return np.sum(np.array(self.heatmap), axis=0)

    def process_image(self, image):
        # NOTE: The output you return should be a color image (3 channel)
        # for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image with lines are drawn on lanes)
        # get image dimensions for slope filter and masking
        if self.first_frame:
            # set the frame flag
            self.first_frame = False
        #print("**FRAME #:", self.frame, " image shape:", image.shape)
        result, self = pipeline(image, self)
        # plt.imshow(result)
        self.frame += 1
        return result
