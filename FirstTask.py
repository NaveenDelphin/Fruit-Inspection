import cv2
import numpy as np

# Load NIR and color images
nir_image = cv2.imread('C0_000002.png', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('C0_000002.png')

# Thresholding to segment the fruit
blurred_nir = cv2.GaussianBlur(nir_image, (5, 5), cv2.BORDER_DEFAULT)
_, thresh = cv2.threshold(blurred_nir, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Fill holes inside the fruit blob
thresh_filled = thresh.copy()
cv2.floodFill(thresh_filled, None, (0, 0), 255)
thresh_filled = cv2.bitwise_not(thresh_filled)

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
