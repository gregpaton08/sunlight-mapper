import torch
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import sys

np.set_printoptions(threshold=sys.maxsize)


def np_stats(input: np.ndarray) -> None:
    print(f"min {np.min(input)} max {np.max(input)} median {np.median(input)}")


# Load the image
# image_filename = "/Users/gpaton/code/frigate-api/pics/backyard-107.jpg"
image_filename = "/Users/gpaton/code/frigate-api/pics/backyard-105.jpg"
# image_filename = "/Users/gpaton/code/frigate-api/pics/backyard-110.jpg"
image = cv2.imread(image_filename)
print(type(image))
scale_factor = 0.25
image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
cv2.imshow("original", image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imshow("RGB", image_rgb)

# Load the baseline image.
baseline_image_filename = "/Users/gpaton/code/frigate-api/pics/backyard-29.jpg"
baseline_image = cv2.imread(baseline_image_filename)
scale_factor = 0.25
baseline_image = cv2.resize(baseline_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
baseline_image_rgb = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2RGB)


def get_value_channel(rgb_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)[:, :, 2]

baseline_value_channel = get_value_channel(baseline_image_rgb)
# cv2.imshow("baseline_value_channel", baseline_value_channel)


# Convert the image to HSV to identify bright regions
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# value_channel = hsv_image[:, :, 2]
value_channel = get_value_channel(image)
# cv2.imwrite("output_value.jpg", value_channel)
# cv2.imshow("HSV Image", value_channel)

diff = cv2.absdiff(baseline_value_channel, value_channel)
np_stats(diff)
normalized_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
np_stats(normalized_diff)
cv2.imshow("diff", normalized_diff)


# Optionally, apply thresholding to find bright regions
# https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
THRESHOLD=64
_, bright_regions = cv2.threshold(normalized_diff, THRESHOLD, 255, cv2.THRESH_BINARY)
# _, bright_regions = cv2.threshold(value_channel, THRESHOLD, 255, cv2.THRESH_BINARY)
cv2.imshow("brightness", bright_regions)




kernel_size = (15, 15)
blurred = cv2.GaussianBlur(bright_regions, kernel_size, 0)
_, bright_regions = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("blurred", bright_regions)


kernel_size = (31, 31)
blurred = cv2.GaussianBlur(bright_regions, kernel_size, 0)
_, bright_regions = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("blurred", bright_regions)



def flatten_image(image: np.ndarray) -> np.ndarray:
    pixels = image.reshape((-1, 1))

    # Optionally, add spatial coordinates (height, width) to the data
    # This can help DBSCAN consider spatial proximity in clustering
    height, width = image.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Combine pixel values and spatial coordinates
    return np.hstack([pixels, coords])


# clustering = DBSCAN(eps=16, min_samples=64).fit(flatten_image(bright_regions))
# # print(type(bright_regions))
# print(f"bright_regions.shape = {bright_regions.shape}")
# # print(type(clustering.labels_))
# print(clustering.labels_.shape)
# height, width = bright_regions.shape
# labels = clustering.labels_.reshape(height, width)
# labels[labels < 0] = 0
# print(labels.shape)
# np_stats(labels)
# labels = labels.astype(np.uint8)
# _, labels = cv2.threshold(labels, 0.5, 255, cv2.THRESH_BINARY)
# cv2.imshow("DBSCAN", labels)







def mask_image(image: np.ndarray, mask: np.ndarray) -> None:
    # Convert the binary mask to a 3-channel mask
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Define the color for the mask overlay (e.g., red)
    mask_color = np.array([0, 0, 255])  # BGR format (red color)

    # Apply the color to the mask
    colored_mask[mask == 255] = mask_color

    # Overlay the mask on the image using bitwise operations
    overlay = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

    cv2.imshow("mask", overlay)


mask_image(image, bright_regions)





# cv2.waitKey(0)
# cv2.destroyAllWindows()
# quit()

# Specify a point in the bright region as a prompt
# You can use the thresholded image to choose a bright point or manually select it
# For simplicity, let's select the brightest point
coords = np.column_stack(np.where(bright_regions > 0))
# print(f"coords = {coords}")
point = coords[0]  # Select the first bright point

# for point in coords:
# print(f"point = {point}")
point = np.array([point])  # SAM expects a numpy array

# # Apply the mask to the image
# sunlight_mask = masks[0].astype(np.uint8) * 255  # Convert boolean mask to binary

# # Show the original image and the mask
# cv2.imshow("Original Image", image)
# cv2.imshow("Sunlight Mask", sunlight_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
