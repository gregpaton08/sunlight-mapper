import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/gpaton/code/frigate-api/pics/backyard-2.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
# https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply Otsu's thresholding
# https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations
kernel = np.ones((5,5),np.uint8)
dilated = cv2.dilate(thresholded, kernel, iterations = 2)
eroded = cv2.erode(dilated, kernel, iterations = 2)

# Show the results
cv2.imshow('Original Image', image)
cv2.imshow('Shadow Segmentation', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
