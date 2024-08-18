import cv2
from skimage.filters import frangi
import numpy as np

# Load the image
image = cv2.imread('muthusada_processed.jpeg', cv2.IMREAD_GRAYSCALE)

# Enhance fingerprint lines using the Frangi filter
enhanced_image = frangi(image)

# Normalize the enhanced image to the range [0, 255]
normalized_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)

# Convert the normalized image to an 8-bit format
converted_image = cv2.convertScaleAbs(normalized_image)

# Use adaptive thresholding to binarize the image
binary_image = cv2.adaptiveThreshold(converted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

# Use morphological operations to clean the binary image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)

# Optionally, use edge detection to extract finer details
edges = cv2.Canny(cleaned_image, 50, 150)

# Combine the edges with the cleaned image
combined_image = cv2.bitwise_and(cleaned_image, cleaned_image, mask=edges)

# Save the extracted fingerprint lines
cv2.imwrite('extracted.jpg', combined_image)

# # Display the results (optional)
# cv2.imshow('Original Image', image)
# cv2.imshow('Enhanced Image', converted_image)
# cv2.imshow('Binary Image', binary_image)
# cv2.imshow('Cleaned Image', cleaned_image)
# cv2.imshow('Extracted Fingerprint Lines', combined_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
