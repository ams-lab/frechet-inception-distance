#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:42:22 2018

@author: tsugaike3
"""

import os, glob
import glob
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.datasets import mnist
from keras.models import Model
from PIL import Image as pil_image
from scipy.linalg import sqrtm

model = InceptionV3() # Load a model and its weights
model4fid = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
def resize_mnist(x):
    x_list = []
    for i in range(x.shape[0]):
        img = image.array_to_img(x[i, :, :, :].reshape(28, 28, -1))
        #img.save("mnist-{0:03d}.png".format(i))
        img = img.resize(size=(299, 299), resample=pil_image.LANCZOS)
        x_list.append(image.img_to_array(img))
    return np.array(x_list)

def resize_do_nothing(x):
    return x

def frechet_distance(m1, c1, m2, c2):
    return np.sum((m1 - m2)**2) + np.trace(c1 + c2 - 2*(sqrtm(np.dot(c1, c2))))

def mean_cov(x):
    mean = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    return mean, sigma

def fid(h1, h2):
    m1, c1 = mean_cov(h1)
    m2, c2 = mean_cov(h2)
    return frechet_distance(m1, c1, m2, c2)

def calc_h(x, resizer, batch_size=8):
    r = None
    n_batch = (x.shape[0]+batch_size-1) // batch_size
    for j in range(n_batch):
        x_batch = resizer(x[j*batch_size:(j+1)*batch_size, :, :, :])
        r_batch = model4fid.predict(preprocess_input(x_batch))
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
        if j % 10 == 0:
            print("i =", j)
    return r

def mnist_h(n_train, n_val):
    x = [0, 0]; h = [0, 0]; n = [n_train, n_val]
    (x[0], _), (x[1], _) = mnist.load_data()
    for i in range(2):
        x[i] = np.expand_dims(x[i], axis=3) # shape=(60000, 28, 28) --> (60000, 28, 28, 1)
        x[i] = np.tile(x[i], (1, 1, 1, 3)) # shape=(60000, 28, 28, 1) --> (60000, 28, 28, 3)
        h[i] = calc_h(x[i][0:n[i], :, :, :], resize_mnist)
    return h[0], h[1]


#def imagenet_h(files, batch_size=8):
#    xs = []; hs = []
#    for f in files:
#        img = image.load_img(f, target_size=(299, 299))
#        x = image.img_to_array(img) # x.shape=(299, 299, 3)
#        xs.append(x)
#        if len(xs) == batch_size:
#            hs.append(calc_h(np.array(xs), resize_do_nothing))
#            xs = []
#    if len(xs) > 0:
#        hs.append(calc_h(np.array(xs), resize_do_nothing))
#    return np.concatenate(hs, axis=0)

# Calculate and save H of MNIST
h_train, h_val = mnist_h(3000, 3000)
np.save("mnist_h_train.npy", h_train)
np.save("mnist_h_val.npy", h_val)

# Calculate and save H of the part of Imagenet 
#h_imagenet = imagenet_h(glob.glob("from_imagenet/*.jpg")) # 10 classes
#h_imagenet_seq = imagenet_h(sorted(glob.glob("from_imagenet_seq/*.jpg"))[0:2956]) # 6 classes
#np.save("imagenet_h.npy", h_imagenet)
#np.save("imagenet_h_seq.npy", h_imagenet_seq)

# Load H and calculate FID
h_train = np.load("mnist_h_train.npy")
h_val = np.load("mnist_h_val.npy")
#h_imagenet = np.load("imagenet_h.npy")
#h_imagenet_seq = np.load("imagenet_h_seq.npy")
print("FID between MNIST train and val :", fid(h_train, h_val))
print("FID between MNIST val and train :", fid(h_val, h_train))
print("FID between MNIST train and train :", fid(h_train, h_train))
#print("FID between MNIST train and imagenet :", fid(h_train, h_imagenet))
#print("FID between MNIST train and imagenet_seq :", fid(h_train, h_imagenet_seq))
#print("FID between imagenet and imagenet_seq :", fid(h_imagenet, h_imagenet_seq))