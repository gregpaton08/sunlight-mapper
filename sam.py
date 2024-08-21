import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np




# checkpoint_file = "/Users/gpaton/code/sunlight-mapper/checkpoint/istd.ckpt"
# checkpoint = torch.load(checkpoint_file, map_location='cpu')

# print(checkpoint)
# quit()


# Load the SAM model
# sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to SAM model checkpoint
sam_checkpoint = "/Users/gpaton/code/sunlight-mapper/checkpoint/istd.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"device = {device}")
model_type = "vit_h"  # Model type (vit_h, vit_b, etc.)



# Load checkpoint as a state dict if it's a PyTorch model
checkpoint = torch.load(sam_checkpoint, map_location=device)


start_epoch = checkpoint['epoch']

state_dict = checkpoint['state_dict']
checkpoint_ = {}
for k, v in state_dict.items():
    if not k.startswith('model.'):
        continue

    k = k[6:] # remove 'model.'
    checkpoint_[k] = v

# Initialize the model and load state dict
sam_model = sam_model_registry[model_type](checkpoint=None)
sam_model.load_state_dict(checkpoint_)
sam_model.to(device)



# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device)

# Load the image
image_filename = "/Users/gpaton/code/frigate-api/pics/backyard-1.jpg"
image = cv2.imread(image_filename)
scale_factor = 0.25
image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the SAM predictor
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Convert the image to HSV to identify bright regions
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
value_channel = hsv_image[:, :, 2]
# cv2.imshow("HSV Image", value_channel)

# Optionally, apply thresholding to find bright regions
# https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
THRESHOLD=200
_, bright_regions = cv2.threshold(value_channel, THRESHOLD, 255, cv2.THRESH_BINARY)
cv2.imshow("brightness", bright_regions)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# quit()

# Specify a point in the bright region as a prompt
# You can use the thresholded image to choose a bright point or manually select it
# For simplicity, let's select the brightest point
coords = np.column_stack(np.where(bright_regions > 0))
print(f"coords = {coords}")
point = coords[0]  # Select the first bright point

# for point in coords:
print(f"point = {point}")
point = np.array([point])  # SAM expects a numpy array

# Generate a mask with SAM
masks, _, _ = predictor.predict(point_coords=coords, point_labels=np.array([1] * len(coords)))
# masks, _, _ = predictor.predict(point_coords=point, point_labels=np.array([1]))

# Apply the mask to the image
sunlight_mask = masks[0].astype(np.uint8) * 255  # Convert boolean mask to binary

# Show the original image and the mask
cv2.imshow("Original Image", image)
cv2.imshow("Sunlight Mask", sunlight_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
