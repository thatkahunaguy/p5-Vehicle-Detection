class Classifier(object):
    """
    Trained classifier
    """

    def __init__(self, classifier, scaler, orient, color_space, pix_per_cell,
                    cell_per_block, spatial_size, hist_bins):
        """
        Initializes an instance.
        Parameters
        ----------
        classifier  : Trained SciPy classifier for detecting vehicles.
        scaler      : SciPy scaler to apply to X.
        """
        self.classifier = classifier
        self.scaler = scaler
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
