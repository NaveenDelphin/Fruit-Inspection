import cv2
import numpy as np

# Load NIR and color images
nir_image = cv2.imread('C0_000002.png', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('C1_000002.png')

# Compute mask on NIR image
_, mask = cv2.threshold(nir_image, 50, 255, cv2.THRESH_BINARY)

# Apply mask to color image
result_img = cv2.bitwise_and(color_image, color_image, mask=mask)

# Convert color image to LAB color space
lab_image = cv2.cvtColor(result_img, cv2.COLOR_BGR2LAB)
l_channel, _, _ = cv2.split(lab_image)

# Adaptive thresholding on L channel
#adaptive_thresh = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
_, thresh = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform morphological operations
kernel = np.ones((5, 5), np.uint8)
morphed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(morphed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
