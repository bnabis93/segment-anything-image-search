import cv2

from utils import *

############################################
# Read sample image
############################################
image = cv2.imread("samples/truck.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


############################################
# SAM(Segmetn-Anything) setting
############################################
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
############################################
# Select image's specific point using SAM(Segmetn-Anything)
############################################
import numpy as np

predictor.set_image(image)
input_point = np.array([[500, 375]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

mask, score, logit = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
h, w = mask.shape[-2:]
mask = mask.reshape(h, w, 1)
# Mask has a 255 or 0 value
mask = (mask * 255).astype(np.uint8)
# Save mask image
cv2.imwrite("mask.png", mask[:, :])
print("shape of mask: ", mask.shape)
contours = cv2.findContours(
    mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
)[0]
cv2.drawContours(mask, contours, -1, (60, 200, 200), 5)
cv2.drawContours(image, contours, -1, (60, 200, 200), 5)

cv2.imwrite("mask2.png", mask)
cv2.imwrite("image2.png", image)
