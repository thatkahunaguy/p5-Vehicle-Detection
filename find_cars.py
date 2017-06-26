import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from get_features import *
from image_helpers import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins, cspace='BGR2YCrCb'):
    #TODO: refactor
    draw_img = np.copy(img)
    # TODO TROUBLESHOOTING - REMOVING THIS LINE AS TEST PER FORUMS
    #img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    # this is just color space conversion and is the image we extract from
    ctrans_tosearch = convert_color(img_tosearch, cspace)
    #print(ctrans_tosearch.shape)
    #print(ctrans_tosearch[0])
    # we now scale the image if needed rather than scaling the windows themselves
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        #print(ctrans_tosearch.shape)
        #print(ctrans_tosearch[0])

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]


    # Compute individual channel HOG features for the entire image
    # this is the computationally intensive piece we only want to do once
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # initialize list for "hot" windows where a car is detected
    on_windows = []

    # possibly deal with differnt scales here
    windows, nblocks_per_window, size = slide_windows(ch1.shape, orient, pix_per_cell, cell_per_block )
    dummy_img = np.copy(img)
    new_window_list = []
    for window in windows:
        new_window_list.append(((window[1], window[0]), (window[1] + 64, window[0] + 64)))
    dummy_img = draw_boxes(dummy_img, new_window_list)

    # now scan the windows across and extract the HOG data as well as color & spatial
    for window in windows:
        ytop = window[0]
        xleft = window[1]
        # Extract HOG for this patch
        ypos = int(ytop/pix_per_cell)
        xpos = int(xleft/pix_per_cell)
        hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
        hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
        hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

        # Extract the image patch
        subimg = cv2.resize(ctrans_tosearch[ytop:ytop+size, xleft:xleft+size], (64,64))

        # Note that we are extracting color & spatial features for each
        # patch since we can't extract for whole image and subsample like HOG
        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)

        # Scale features and make a prediction
        test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        # use the classifier to make a prediction
        test_prediction = svc.predict(test_features)
        # take action for "hot" windows where a car was predicted
        # now we need to scale the dimensions since the image was scaled
        # note that we are drawing and passing back an image rather than
        # a list of boxes with hits as we previously did
        if test_prediction == 1:
            xbox_left = np.int(xleft*scale)
            #print("hot xleft xbox_left:", xleft, xbox_left)
            ytop_draw = np.int(ytop*scale)
            #print("hot ytop, ytop_draw:", ytop, ytop_draw)
            win_draw = np.int(size*scale)
            on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
            cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
    #print("on_windows:", on_windows)
    return on_windows

def slide_windows(shape, orient, pix_per_cell, cell_per_block):
    #TODO: refactor
    windows = []
    #print("shape:", shape)
    # Define blocks and steps as above
    nxblocks = (shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (shape[0] // pix_per_cell) - cell_per_block + 1
    #print("blocks: nyblocks, nxblocks", nyblocks, nxblocks)
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    # This is defining the window size for a scale of 1 and is also the
    # size the SVC was trained with so must be fixed in this case since
    # we are not training the SVC here
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #print("blocks: nblocks_per_window", nblocks_per_window)
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    #print("step: ", nysteps, nxsteps)
    # set ytop at 0 in case there are no y steps across the window
    ytop = 0
    for xb in range(nxsteps):
        xleft = xb*cells_per_step* pix_per_cell
        if nysteps == 0: windows.append((ytop, xleft))
        for yb in range(nysteps):
            ytop = yb*cells_per_step*pix_per_cell
            windows.append((ytop, xleft))
    return windows, nblocks_per_window, window
