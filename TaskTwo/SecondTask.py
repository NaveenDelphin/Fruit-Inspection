import cv2
import numpy as np

# Load color and NIR images
color_img = cv2.imread('C1_000004.png')
nir_img = cv2.imread('C0_000004.png', 0)

cv2.imshow("clrimg",color_img)

# Remove black background from color image
color_img[nir_img == 0] = [0, 0, 0]

# Convert color image to HSV
hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

'''
# Define yellow color range in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Threshold the HSV image to extract yellow regions
yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
'''

# Calculate Mahalanobis distance
def mahalanobis_distance(color, mean_color, cov_inv):
    diff = color - mean_color
    return np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

'''
#First 000005
# Define mean color and covariance matrix of russet color in HSV space
russet_mean = np.array([18, 139, 133])  # Specify mean hue (h), saturation (s), and value (v)
russet_cov_inv = np.linalg.inv([[  1.17262904 , -0.605892 ,    1.01041996],[ -0.605892  ,  34.8665849  ,-25.65092306],[  1.01041996 ,-25.65092306 , 84.7698647 ]])  # Specify the inverse of the covariance matrix
'''


#Second 000004
# Define mean color and covariance matrix of russet color in HSV space
russet_mean = np.array([13, 139, 30])  # Specify mean hue (h), saturation (s), and value (v)
russet_cov_inv = np.linalg.inv([[8.50980513  ,10.26967436  ,17.92880256],[ 10.26967436 ,191.89524615  ,74.56542564],[ 17.92880256 , 74.56542564 ,113.52765641]])  # Specify the inverse of the covariance matrix


# Calculate Mahalanobis distance for each pixel
mahalanobis_distances = np.zeros_like(hsv_img[:, :, 0], dtype=np.float32)
for i in range(hsv_img.shape[0]):
    for j in range(hsv_img.shape[1]):
        pixel_color = hsv_img[i, j, :]
        mahalanobis_distances[i, j] = mahalanobis_distance(pixel_color, russet_mean, russet_cov_inv)

# Threshold the Mahalanobis distances to detect russet regions
russet_threshold = 10  # Adjust this threshold as needed
russet_mask = mahalanobis_distances > russet_threshold

# Perform morphological operations to refine the detected regions if necessary
kernel = np.ones((5, 5), np.uint8)
russet_mask = cv2.morphologyEx(russet_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

# Apply the russet mask to the color image
russet_detected_img = cv2.bitwise_and(color_img, color_img, mask=russet_mask)

# Display the result
cv2.imshow('Russet Detection', russet_detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
