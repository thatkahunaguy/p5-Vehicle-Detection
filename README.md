# **Vehicle Detection & Tracking - Project #5** 

### John Glancy

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

**Included Files**
1) Project Writeup: [P5_vehicle_detection.html](/P5_vehicle_detection.html)
2) [P5_vehicle_detection.ipynb](/P5_vehicle_detection.ipynb): Jupyter notebook with the project
3) Python Files Imported to Notebook in 2:
* [VehicleTracker.py](/VehicleTracker.py): class to track vehicle heatmap info across frames		
* [find_cars.py](/find_cars.py): sliding window and scaling to extract features and predict cars with classifier
* [get_features.py](/get_features.py): HOG/color/spatial feature extraction
* [image_helpers.py](/image_helpers.py):routines to calculate, threshold & render heatmaps, bounding boxes, & overlays
* [Classifier.py](/Classifier.py): class to store Classifier information	
* [camera_prep.py](/camera_prep.py): routines to calibrate the camera
* [pipeline.py](/pipeline.py): processing pipeline for images
* [plot_images.py](/plot_images.py): helper for plotting
4) [output images folder](/output_images): example test output images
5) [test_image_output](/test_image_output) folder: pipeline output for all test images
6) [project_output_video.mp4](/project_output_video.mp4): final project output video
