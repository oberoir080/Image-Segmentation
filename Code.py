import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Load the image
img = cv2.imread('peppers.png')

# Reshape the image to a 2D array of pixels
img_flat = np.reshape(img, (-1, 3))

# Estimate the bandwidth parameter
bandwidth = estimate_bandwidth(img_flat, quantile=0.1, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(img_flat)
labels = ms.labels_

# Giving random colors to each label
label_colors = np.random.randint(0, 255, (np.max(labels) + 1, 3))

# Creating the coloured image
segmented_img = label_colors[labels]
segmented_img = np.reshape(segmented_img, img.shape)

segmented_img = segmented_img.astype(np.uint8)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()