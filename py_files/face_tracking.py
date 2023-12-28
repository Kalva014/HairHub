from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os
import numpy as np

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        x2 = x1 + w1
        y2 = y1 + h1

        # Load the overlay image
        img2 = cv2.imread(os.path.join(os.getcwd(), "filters/hairstyle1.png"))

        # Manually set a larger resizing factor (adjust as needed)
        resizing_factor = 2.5
        new_w = int(img2.shape[1] * resizing_factor)
        new_h = int(img2.shape[0] * resizing_factor)

        img2_resized = cv2.resize(img2, (new_w, new_h))

        # Create a mask based on one of the color channels (e.g., red channel)
        mask = img2_resized[:, :, 2]  # Use the red channel for the mask

        # Invert the mask (black becomes white, white becomes black)
        mask_inv = cv2.bitwise_not(mask)

        # Move the filter up by adjusting the ROI coordinates
        roi_y1 = max(0, y1 - int(0.5 * h1))  # Move up by 50% of face height
        roi_y2 = roi_y1 + new_h  # Use the new height after resizing

        # Ensure ROI dimensions match the resized overlay image
        roi_h, roi_w = roi_y2 - roi_y1, x2 - x1
        img2_resized = cv2.resize(img2_resized, (roi_w, roi_h))

        # Create a region of interest (ROI) on the original image
        roi = img[roi_y1:roi_y1 + roi_h, x1:x1 + roi_w]

        # Ensure ROI dimensions match the resized overlay image
        roi = cv2.resize(roi, (roi_w, roi_h))

        # Combine the two images with transparency
        img[roi_y1:roi_y1 + roi_h, x1:x1 + roi_w] = cv2.addWeighted(roi, 1, img2_resized, 1, 0)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
