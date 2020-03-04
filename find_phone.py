#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
import tensorflow as tf

def get_center_norm(bbox, img_w, img_h):
    x1,x2,y1,y2 = bbox
    x_i = x1 + (x2-x1)/2.0
    y_i = y1 + (y2-y1)/2.0
    
    return [x_i/img_w, y_i/img_h]

image_path = sys.argv[1]
img = cv2.imread(image_path)
img = img.astype(np.uint8)

img_h = 224
img_w = 224 

img = cv2.resize(img, (img_w,img_h))

model = tf.keras.models.load_model('rcnn_model')

num_regions = 200 # Cap on number of predicted regions

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() # selective search

ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
ss_results = ss.process()
img_out = img.copy()

X_pred = []
y_pred = []

for e, res in enumerate(ss_results):
    if e < num_regions:
        x1,y1,w,h = res #get predicted bbox
        pred_img = img_out[y1:y1+h,x1:x1+w]
        resized = cv2.resize(pred_img, (img_w,img_h))
        X_pred.append(resized)
        y_pred.append([x1, x1+w, y1, y1+h])
    else:
        break

confidence_scores = model.predict(np.asarray(X_pred))

#In this case we have one phone per image, however this could be thresholded to include multiple instances
idx_max = np.where(confidence_scores[:,0] == max(confidence_scores[:,0]))[0][0]

bbox = y_pred[idx_max]
pred = get_center_norm(bbox, img_w, img_h)

print(pred)