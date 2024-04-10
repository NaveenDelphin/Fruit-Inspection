import cv2
import numpy as np

def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
    return dist

# Load color and NIR images
color_img = cv2.imread('C1_000005.png')
nir_img = cv2.imread('C0_000005.png', 0)
color_img[nir_img == 0] = [0, 0, 0]

# Define the region of interest (ROI) for kiwi fruits
roi = cv2.selectROI("Select ROI", color_img)
cv2.destroyAllWindows()

# Crop the ROI from the original image
kiwi_roi = color_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# Convert the ROI to the LAB color space
kiwi_roi_lab = cv2.cvtColor(kiwi_roi, cv2.COLOR_BGR2LAB)

# Calculate the mean and covariance matrix of LAB channels
mean_lab = np.mean(kiwi_roi_lab, axis=(0, 1))
cov_lab = np.cov(kiwi_roi_lab.reshape(-1, 3).T)

# Compute the Mahalanobis distance for each pixel
height, width = color_img.shape[:2]
mahalanobis_img = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        pixel_lab = cv2.cvtColor(np.uint8([[color_img[y, x]]]), cv2.COLOR_BGR2LAB)[0][0]
        mahalanobis_img[y, x] = mahalanobis_distance(pixel_lab, mean_lab, cov_lab)

# Threshold the Mahalanobis distances
threshold = 4  # Adjust threshold as needed
russet_mask = mahalanobis_img > threshold

# Perform morphological operations to refine the detected regions if necessary
kernel = np.ones((5, 5), np.uint8)
russet_mask = cv2.morphologyEx(russet_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

# Apply the russet mask to the color image
russet_detected_img = cv2.bitwise_and(color_img, color_img, mask=russet_mask)

# Display the result
cv2.imshow('Russet Detection', russet_detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()