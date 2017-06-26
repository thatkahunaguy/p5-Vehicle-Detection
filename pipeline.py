# import image search and sliding window routines for car classification
from find_cars import *
# import the helpers to draw bounding boxes, overlays etc
from image_helpers import *
# pipeline for VehicleTracker object
def pipeline(img, cars):
    out_win = []
    # 528
    # prior 464 start for 1.5, 2, 3
    out_win = find_cars(img, 400, 528, 1, cars.svc, cars.X_scaler, cars.orient, cars.pix_per_cell,
                        cars.cell_per_block, cars.spatial_size, cars.hist_bins, cars.cspace)
    out_win += find_cars(img, 464, 656, 2, cars.svc, cars.X_scaler, cars.orient, cars.pix_per_cell,
                         cars.cell_per_block, cars.spatial_size, cars.hist_bins, cars.cspace)
    #out_win += find_cars(img, 400, 656, 2.5, cars.svc, cars.X_scaler, cars.orient, cars.pix_per_cell,
    #                     cars.cell_per_block, cars.spatial_size, cars.hist_bins, cars.cspace)
    out_win += find_cars(img, 400, 528, 1.5, cars.svc, cars.X_scaler, cars.orient, cars.pix_per_cell,
                         cars.cell_per_block, cars.spatial_size, cars.hist_bins, cars.cspace)
    out_win += find_cars(img, 400, 528, 1.2, cars.svc, cars.X_scaler, cars.orient, cars.pix_per_cell,
                         cars.cell_per_block, cars.spatial_size, cars.hist_bins, cars.cspace)
    heatmap = np.zeros_like(img[:,:,0])
    # adjust the threshold for early frames
    #current_thresh = min(len(cars.heatmap)+2, cars.threshold)
    current_thresh = cars.threshold
    # create a heatmap from the hot windows
    heatmap = add_heat(heatmap, out_win)
    # label and increase heat on intersecting bounding boxes
    # identify initial labels including false positives
    labels = label(heatmap)
    # add heat to the hot windows which intersect each other to eliminate false positives
    # and retain vehicle edges - note this keeps too much of prior frames vehicle location
    heatmap2 = increase_heat(np.copy(heatmap), labels, current_thresh)
    # add the increased heat heatmap to the deque
    cars.heatmap.append(heatmap2)
    # saving original heatmap for debug vs. increased heat
    #mpimg.imsave("output_images/heatmap" + str(cars.frame) + ".jpg", heatmap)
    #
    heatmap_sum = cars.get_heatmap_sum()
    # test the scan bounding boxes routine
    #heatmap_sum1 = scan_bounding_boxes(cars, heatmap_sum, 8, 16, 5)
    #mpimg.imsave("output_images/heatmap_sum" + str(cars.frame) + ".jpg", heatmap_sum)
    # heatmap2 = heatmap_sum
    # threshold the heatmap sum
    heatmap3 = apply_threshold(np.copy(heatmap_sum), current_thresh)
    #mpimg.imsave("output_images/heatmap_thresh" + str(cars.frame)+".jpg", heatmap3)
    cars.thresh_heatmap.append(heatmap3)
    plt.imshow(heatmap, cmap='hot')
    # label the thesholded image to identify bounding boxes
    labels_thresh = label(heatmap3)
    labeled_boxes, bboxes = draw_labeled_bboxes(np.copy(img), labels_thresh, ret_boxes=True)
    cars.bounding_boxes.append(bboxes)
    # add the heatmaps to the labeled_boxes image
    h, w = 240, 426
    add_overlay(heatmap, labeled_boxes, 0, h, w)
    add_overlay(heatmap_sum, labeled_boxes, 1, h, w)
    add_overlay(heatmap3, labeled_boxes, 2, h, w)
    labeled_boxes = cv2.addWeighted(labeled_boxes, 1, cars.frame_label, 0.3, 0)
    return labeled_boxes, cars
