#! /usr/bin/python
from DataGenerator import DataGenerator
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import datasets, layers, Model, utils, optimizers, callbacks
from tensorflow.keras.applications.vgg16 import VGG16

#Class to define the on_train_batch_end method in callbacks to terminate training at a certain accuracy.
class accCallback(callbacks.Callback):

	def on_train_batch_end(self, batch, logs={}):
		if(logs.get('accuracy') > 0.96):
			self.model.stop_training = True

#Method to obtain epsilon bounding boxes for center point
def gen_bbox(x, y, epsilon_w, epsilon_h):
    assert epsilon_w % 2 == 0
    assert epsilon_h % 2 == 0
    
    x1 = x - epsilon_w/2
    x2 = x + epsilon_w/2
    y1 = y - epsilon_h/2
    y2 = y + epsilon_h/2
    
    return [int(x1), int(x2), int(y1), int(y2)]

#Get the intersection over union for two bounding boxes
def get_IOU(bbox1, bbox2):
    #Get intersection points
    x_left = max(bbox1[0], bbox2[0])
    x_right = min(bbox1[1], bbox2[1])
    y_top = max(bbox1[2], bbox2[2])
    y_bot = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bot < y_top: 
        return 0.0
    
    bbox1_area = (bbox1[1] - bbox1[0])*(bbox1[3] - bbox1[2])
    bbox2_area = (bbox2[1] - bbox2[0])*(bbox2[3] - bbox2[2])
    int_area = (x_right - x_left) * (y_bot - y_top)
    
    iou = int_area/(float(bbox1_area) + float(bbox2_area) - float(int_area))
    
    return iou

#Get folder path (command line argument)
dataset = sys.argv[1]

with open(os.path.join(dataset,'labels.txt')) as inFile:
    labelData=inFile.readlines()

#Size for VGG16 inputs
img_h = 224
img_w = 224 

X = []
y = []

for line in labelData:
    imName, x_i, y_i = line.split(' ')
    img = cv2.imread(os.path.join(dataset, imName))
    #img = cv2.imread(os.path.join(dataset, imName))
    X.append(cv2.resize(img, (img_w,img_h)))
    y.append([float(x_i)*img_w,float(y_i)*img_h])
X = np.asarray(X)
y = np.asarray(y)

data = X
labels = y #Ensure that the centers are transformed appropriately
datagen = DataGenerator(data, labels)
datagen.set_probabilities([1, 0.8, 0.3]) #set probabilities for shift, rotate, noise
X_a, y_a = datagen.generate([datagen.shift, datagen.rotate],len(data)) #duplicate data

X_train = np.concatenate((X, X_a))
y_train = np.concatenate((y,y_a))

#Shuffle training data
idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]

epsilon_w = 16
epsilon_h = 16

y_bbox = []

for i in range(len(X_train)):
    x_i, y_i = y_train[i]
    bbox = gen_bbox(int(x_i),int(y_i), epsilon_w, epsilon_h) #convert to pixel values
    y_bbox.append(np.asarray(bbox))
    
y_train = np.asarray(y_bbox) #replace y_train with y_bbox

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() # selective search

#loop through images
X_new = []
y_new = []
ious = []

for i in tqdm(range(len(X_train)), desc='Generating Predicted ROIs'):

    #Convert image to type uint - only type compatible with ss
    img = X_train[i]
    img = img.astype(np.uint8)
    bbox = y_train[i]

    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    img_out = img.copy()
    counter = 0
    falsecounter = 0
    flag = 0
    fflag = 0
    bflag = 0
    for e, res in enumerate(ss_results):
        if e < 2000 and flag == 0: 
            x1,y1,w,h = res #get predicted bbox

            iou = get_IOU(bbox, [x1, x1+w, y1, y1+h])
            ious.append(iou)

            #collect maximum of 30 pos samples
            if counter < 30: 
                if iou > 0.70:
                    train_img = img_out[y1:y1+h,x1:x1+w]
                    resized = cv2.resize(train_img, (img_w,img_h))
                    X_new.append(resized)
                    one_hot  = [1,0] #positive
                    y_new.append(one_hot)
                    counter+=1
            else:
                fflag = 1
            if falsecounter < 30:
                if iou < 0.3:
                    train_img = img_out[y1:y1+h,x1:x1+w]
                    resized = cv2.resize(train_img, (img_w,img_h))
                    X_new.append(resized)
                    one_hot = [0,1] #negative
                    y_new.append(one_hot)
                    falsecounter+=1
            else:
                bflag = 1
            if fflag == 1 and bflag == 1:
                print('inside')
                flag=1
                        
X_new = np.asarray(X_new)
y_new = np.asarray(y_new)              

#Train last layer of VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)

#Freeze first 15 layers
for layer in (vggmodel.layers)[:15]:
    layer.trainable=False

#replace last layer with dense
fc_out= vggmodel.layers[-2].output
fc_out = layers.Dense(2, activation="softmax")(fc_out)

callback = accCallback()

model = Model(inputs=vggmodel.input, outputs=fc_out)
model.compile(loss = tf.keras.losses.categorical_crossentropy, 
              optimizer=optimizers.Adam(lr=0.0001), 
              metrics = ['accuracy'])

history = model.fit(X_new, y_new, epochs=1, batch_size=16, 
                    validation_split=0.2, callbacks=[callback])
model.save('rcnn_model')
print('Model Saved')