import cv2
import numpy as np

def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
    return dist

# Load the color image
image = cv2.imread('C1_000006.png')
nir_image = cv2.imread('C0_000006.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(nir_image, 50, 255, cv2.THRESH_BINARY)

# Apply mask to color image
result_img = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("masimg",result_img)

# Define the region of interest (ROI) for kiwi fruits
roi = cv2.selectROI("Select ROI", result_img)
cv2.destroyAllWindows()

# Crop the ROI from the original image
kiwi_roi = result_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# Convert the ROI to the LAB color space
kiwi_roi_lab = cv2.cvtColor(kiwi_roi, cv2.COLOR_BGR2LAB)
cv2.imshow("cvrt",kiwi_roi_lab)
# Calculate the mean and covariance matrix of LAB channels
mean_lab = np.mean(kiwi_roi_lab, axis=(0, 1))
cov_lab = np.cov(kiwi_roi_lab.reshape(-1, 3).T)

# Compute the Mahalanobis distance for each pixel
height, width = result_img.shape[:2]
mahalanobis_img = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        pixel_lab = cv2.cvtColor(np.uint8([[image[y, x]]]), cv2.COLOR_BGR2LAB)[0][0]
        mahalanobis_img[y, x] = mahalanobis_distance(pixel_lab, mean_lab, cov_lab)

# Threshold the Mahalanobis distances
threshold = 5  # Adjust threshold as needed
mask = np.where(mahalanobis_img < threshold, 255, 0).astype(np.uint8)

# Apply the mask to the original image
result = cv2.bitwise_and(result_img, result_img, mask=mask)
cv2.imshow("rs",result)

result[nir_image == 0] = [0, 0, 0]
grayImage = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyimg",grayImage)

# Thresholding to segment the fruit
blurred_nir = cv2.GaussianBlur(grayImage, (5, 5), cv2.BORDER_DEFAULT)
_, thresh = cv2.threshold(blurred_nir, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Fill holes inside the fruit blob
thresh_filled = thresh.copy()
cv2.floodFill(thresh_filled, None, (0, 0), 255)
thresh_filled = cv2.bitwise_not(thresh_filled)
cv2.imshow("thresh_filled",thresh_filled)

#Masking the original color image

# Erosion followed by Minkowsky subtraction
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(thresh_filled, kernel, iterations=1)
minkowski_subtraction = cv2.subtract(thresh_filled, eroded)

# Find contours
contours, _ = cv2.findContours(minkowski_subtraction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on color image
for contour in contours:
    # Calculate center and radius of the minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    # Draw the circle around the contour
    cv2.circle(result, center, radius, (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
