import torch
# from segment_anything import sam_model_registry
from build_sam import sam_model_registry
import cv2
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = "cpu"
# MODEL_TYPE = "vit_h"
MODEL_TYPE = "vit_b"
# CHECKPOINT_PATH="./sam_vit_h_4b8939.pth"
CHECKPOINT_PATH="./checkpoint/sam/sam_vit_b_01ec64.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
# sam.to(device=DEVICE)

checkpoint_file = "/Users/gpaton/code/sunlight-mapper/checkpoint/istd.ckpt"
device = torch.device('cpu')
checkpoint = torch.load(checkpoint_file, map_location=device)

state_dict = checkpoint['state_dict']
checkpoint_ = {}
for k, v in state_dict.items():
    if not k.startswith('model.'):
        continue

    k = k[6:] # remove 'model.'
    checkpoint_[k] = v

print("load state dict...")
print(checkpoint_.keys())
sam.load_state_dict(checkpoint_, strict=False)
sam.to(device=DEVICE)
print("loaded` state dict")

mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_PATH = "./IMG_3409.jpg"
image_bgr = cv2.imread(IMAGE_PATH)
scale_factor = 0.125
image_bgr = cv2.resize(image_bgr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print(f"img shape = {image_rgb.shape}")
result = mask_generator.generate(image_rgb)
print(f"finished generating mask. num results = {len(result)}")

# Annotate the original image with the segmentations.
# mask_annotator = sv.MaskAnnotator()
# mask_annotator = sv.MaskAnnotator(color_map = "index")
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
print("finished generating annotation")
detections = sv.Detections.from_sam(result)
print("finished detections")
annotated_image = mask_annotator.annotate(image_bgr, detections)
cv2.imwrite("./output.jpg", annotated_image)
cv2.imshow("Mask", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("done")
