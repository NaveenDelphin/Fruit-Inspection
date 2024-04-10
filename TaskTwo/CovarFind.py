import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load sample image containing russet areas
sample_img = cv2.imread('C1_000004.png')
sample_img_hsv = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
plt.imshow(sample_img_hsv)
plt.show()

'''
#Image 000005
# Select sample region coordinates (e.g., top-left and bottom-right corners)
x1, y1 = 88, 73  # Coordinates of top-left corner
x2, y2 = 145, 114  # Coordinates of bottom-right corner
'''
#Image 000004
# Select sample region coordinates (e.g., top-left and bottom-right corners)
x1, y1 = 120, 190  # Coordinates of top-left corner
x2, y2 = 145, 215 # Coordinates of bottom-right corner

# Extract the sample region
sample_region = sample_img_hsv[y1:y2, x1:x2]

# Calculate covariance matrix
covariance_matrix = np.cov(sample_region.reshape(-1, 3).T)

print("Covariance Matrix:")
print(covariance_matrix)
