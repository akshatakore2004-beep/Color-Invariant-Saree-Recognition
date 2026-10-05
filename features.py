import cv2, numpy as np
from skimage.feature import local_binary_pattern, hog

def extract(path, size=224):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    lbp_h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    h = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.concatenate([lbp_h, h])
