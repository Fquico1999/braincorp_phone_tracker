#! /usr/bin/python
"""
The DataGenerator class takes a dataset of images and labels corresponding to a point on the image 
and applies basic data augmentation transformations. It transforms the labels accordingly.
"""

import numpy as np
import cv2

class DataGenerator():
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.t_min = 0
        self.t_max = 20
        self.h = self.X[0].shape[0]
        self.w = self.X[0].shape[1]
        self.gauss_mu = 0
        self.gauss_var_min = 0.0004
        self.gauss_var_max = 0.008
        self.sp_s_v_p_min = 0.2
        self.sp_s_v_p_max = 0.8
        self.sp_amount_min = 0.001
        self.sp_amount_max = 0.008
        self.prob_shift = 1
        self.prob_rotate = 1
        self.prob_noise = 1
        
    def shift(self,params):
        img, label = params
        
        #Obtain random number from unifrom probability distribution
        prob = np.random.uniform()
        
        if prob <= self.prob_shift:
            #Create translation vector and matrix
            t = np.random.randint(self.t_min,self.t_max,size=(label.shape))
            M = np.float32([[1,0,t[0]],[0,1,t[1]]])
            #obtain shifted image
            dst = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
            #obtain new label
            new_label = label+t
        else:
            dst = img
            new_label = label
        return dst, new_label

    def rotate(self,params):
        img, label = params

        #Obtain random number from unifrom probability distribution
        prob = np.random.uniform()
        if prob <= self.prob_rotate:
            #Simple rotation, random chance of flipping lr or ud.
            p = np.random.randint(0,2)
            if p:
                dst = np.fliplr(img)
                new_label = [abs(self.w-label[0]),label[1]]
            else:
                dst = np.flipud(img)
                new_label = new_label = [label[0],abs(self.h-label[1])]
        else:
            dst = img
            new_label = label
        return dst, new_label
    
    def noise(self,params):
        img, label = params
        #Obtain random number from unifrom probability distribution
        prob = np.random.uniform()
        
        if prob<=self.prob_noise:        
            #random chance of gaussion noise or salt and pepper
            p = np.random.randint(0,2)
            if p == 0:
                #Choose random variation
                var = np.random.uniform(self.gauss_var_min, self.gauss_var_max)
                gauss = np.random.normal(self.gauss_mu, var**0.5, img.shape)
                noise = gauss.reshape(img.shape)
                dst = img+noise
            elif p == 1:
                sp_amount = np.random.uniform(self.sp_amount_min, self.sp_amount_max)
                sp_s_v_p = np.random.uniform(self.sp_s_v_p_min, self.sp_s_v_p_max)
                noise = np.copy(img)
                num_salt = np.ceil(sp_amount * img.size * sp_s_v_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
                noise[coords] = 1
                num_pepper = np.ceil(sp_amount* img.size * (1. - sp_s_v_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
                noise[coords] = 0
                dst = noise
        else:
            dst = img
        return dst, label
    
    def generate(self,aug_f,n_samples):
        aug_X = []
        aug_y = []
        
        for _ in range(n_samples):
            idx = np.random.randint(0, len(self.X))
            y_i = self.y[idx]
            X_i = self.X[idx]
            
            X_new = X_i
            y_new = y_i
            
            for f in aug_f:
                X_new, y_new = f([X_new, y_new])
            
            aug_X.append(X_new)
            aug_y.append(y_new)
        return np.asarray(aug_X), np.asarray(aug_y)
            
    def _set_t(self, t_min, t_max):
        self.t_min = t_min
        self.t_max = t_max
    
    def set_probabilities(self, prob_arr):
        self.prob_shift = prob_arr[0]
        self.prob_rotate = prob_arr[1]
        self.prob_noise = prob_arr[2]