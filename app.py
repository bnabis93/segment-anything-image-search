"""Opencv UI for selecting a region of interest
Author : HyeonWoo Jeong
References
    - https://stackoverflow.com/questions/49799057/how-to-draw-a-point-in-an-image-using-given-co-ordinate-with-python-opencv
"""
import argparse

import cv2
import numpy as np

######################################################################
# Argument parse. Set the image path
######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
args = parser.parse_args()
select_point = []


############################################
# SAM(Segmetn-Anything) setting
############################################
from segment_anything import SamPredictor, sam_model_registry

image = cv2.imread(args.image)
clone = image.copy()
sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)


######################################################################
# load the image, clone it, and setup the mouse callback function
######################################################################
color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)


def select_point_callback(event, x, y, flags, param):
    """
    Select a point in the image
    """
    global select_point
    if event == cv2.EVENT_LBUTTONDOWN:
        select_point = [(x, y)]

        input_point = np.array([[x, y]])
        input_label = np.array([1])
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

        # Point update
        cv2.circle(image, select_point[0], 5, (0, 0, 255), 5)
        cv2.imshow("image", image)

        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        # cv2.drawContours(mask, contours, -1, (60, 200, 200), 5)
        cv2.drawContours(image, contours, -1, (60, 200, 200), 5)

        # cv2.imwrite("mask2.png", mask)
        cv2.imshow("image", image)


cv2.namedWindow("image")
cv2.setMouseCallback("image", select_point_callback)

# keep looping until the 'q' key is pressed
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# close all open windows
cv2.destroyAllWindows()
