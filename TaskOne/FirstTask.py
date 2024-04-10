import cv2
import numpy as np

# Load NIR and color images
nir_image = cv2.imread('C0_000001.png', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('C1_000001.png')

# Compute mask on NIR image
# Example: thresholding the NIR image to create a binary mask
_, mask = cv2.threshold(nir_image, 50, 255, cv2.THRESH_BINARY)

# Apply mask to color image
result_img = cv2.bitwise_and(color_image, color_image, mask=mask)
cv2.imshow("masimg",result_img)
grayImage = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyimg",grayImage)


# Thresholding to segment the fruit
blurred_nir = cv2.GaussianBlur(grayImage, (5, 5), cv2.BORDER_DEFAULT)
#adaptive_thresh = cv2.adaptiveThreshold(blurred_nir, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

_, thresh = cv2.threshold(blurred_nir, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("tresh",thresh)

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
    cv2.circle(color_image, center, radius, (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
