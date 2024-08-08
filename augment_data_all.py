# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import struct
import cv2
import scipy
from skimage import transform
from skimage.transform import AffineTransform
from skimage.morphology import skeletonize, medial_axis
from skimage import exposure
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
import torch
import sys
path_utils = "C:/Documents/Université/Master/Mémoire/Code/example/utils"
if(not(path_utils in sys.path)):
    sys.path.insert(0,path_utils)
input_path = 'C:/Documents/Université/Master/Mémoire/Code/example/MNIST'
param_path = 'C:/Documents/Université/Master/Mémoire/Code/example/parameters'
import mnist_utils as mnu
import torch
import os
import pickle
from tqdm import tqdm
import json
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

data = {'mean': 0.8, 'std':0.2, 'normal': False,
        'sigma1' : 0.55, 'sigma2' : 0.85,'truncate_gaussianFilter': 4.0, 'radius_gaussianFilter': 2,
        'a_translation': 8,'b_translation': 2, 'num_train_samples_translation' : 0, 'num_test_samples_translation': 0,
        'x_shear': 0.3, 'y_shear': 0.2, 'num_train_samples_shear' : 0, 'num_test_samples_shear' : 0,
        'angle_rotation': 10, 'num_train_samples_rotation' : 0 , 'num_test_samples_rotation' : 0,
        'num_train_samples_lines' : 0, 'num_test_samples_lines' : 0}


#%%
mnist_dataloader = mnu.MnistDataloader(training_images_filepath, 
                                   training_labels_filepath, 
                                   test_images_filepath, 
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
num_samples_train = len(x_train)
num_samples_test = len(x_test)

digits_train = np.array(y_train, dtype = 'int')
digits_test = np.array(y_test, dtype = 'int')

# y_train = np.zeros((num_samples_train, 10), dtype = 'float32')
# for k in range(10):
#     mask = digits_train == k
#     y_train[mask,k] = 1

# y_test = np.zeros((num_samples_test, 10), dtype = 'float32')
# for k in range(10):
#     mask = digits_test == k
#     y_test[mask,k] = 1



#%%
num_samples_valid = 10000
x_train = np.array(x_train, dtype = 'float32')
y_train = np.array(y_train)
x_valid = x_train[-num_samples_valid:,...]
y_valid = y_train[-num_samples_valid:,...]
x_train = x_train[:-num_samples_valid,...]
y_train = y_train[:-num_samples_valid,...]
x_test = np.array(x_test, dtype = 'float32')
y_test = np.array(y_test)

num_samples_train = len(x_train)
num_samples_valid = len(x_valid)
num_samples_test = len(x_test)


#%% Normalize and inverse contrast
x_train = mnu.min_max(x_train, axis = (1,2) )
x_valid = mnu.min_max(x_valid, axis = (1,2) )
x_test = mnu.min_max(x_test, axis = (1,2) )

#%%
plt.figure()
plt.imshow(x_train[12,...].squeeze(), cmap = 'binary_r')
plt.colorbar()
plt.show()


#%% Skeleton MNIST
def skeleton(u,mean,std,normal):
    skeletons = np.zeros(u.shape)
    for i in tqdm(range(len(u))):
        skeletons[i]= skeletonize(u[i],method="lee")
        #ke, dist = medial_axis(u[i],return_distance=True)
        #skeletons[i] = dist*ske
    if normal == True :
        random_pixels = np.random.normal(mean,std,np.sum(skeletons>0))
        random_pixels[random_pixels>1]=1
        random_pixels[random_pixels<0]=0
        skeletons[skeletons>0] = random_pixels
    return skeletons

skeleton_x_train = skeleton(x_train,data['mean'],data['std'],data['normal'])
skeleton_x_test = skeleton(x_test,data['mean'],data['std'],data['normal'])

#%%

idx = np.random.choice(len(x_train))
plt.figure()
plt.imshow(skeleton_x_train[idx].squeeze(), cmap = 'gray')
plt.title('Sample : {0} | Label : {1}'.format( idx, str(y_train[idx])))
plt.colorbar()
plt.show()

#%%
first_start = 2
second_start = 18
first_inter_length = 10 
second_inter_length = 8

digits_pairs = [(0,-1), (1,-1), (2,-1), (3,-1), (4,-1), (5,-1), 
                 (6,-1), (7,-1), (8,-1), (9,-1), (1,0), (1,1), (1,2),
                 (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)]

num_new_digits = 10000
all_new_x_train = np.zeros((len(digits_pairs)*num_new_digits, 28 , 54), dtype = 'float32')
new_y_train = np.zeros(len(digits_pairs)*num_new_digits, dtype = 'int')
for j,digit_pair in enumerate(digits_pairs):
    first_digit, second_digit = digit_pair
    all_new_x_train[j*num_new_digits:(j+1)*num_new_digits,...] = mnu.create_new_digits(skeleton_x_train, y_train, 
                                first_digit, second_digit, num_new_digits, 
                                  first_start = first_start, second_start = second_start, 
                                  first_inter_length = first_inter_length, second_inter_length = second_inter_length)
    
    if(second_digit>=0):
        new_y_train[j*num_new_digits:(j+1)*num_new_digits,] = second_digit + 10*first_digit
    else:
        new_y_train[j*num_new_digits:(j+1)*num_new_digits,] = first_digit
    

num_new_digits = 2000
all_new_x_test = np.zeros((len(digits_pairs)*num_new_digits, 28 , 54), dtype = 'float32')
new_y_test = np.ones(len(digits_pairs)*num_new_digits, dtype = 'int')*10
for j,digit_pair in enumerate(digits_pairs):
    first_digit, second_digit = digit_pair
    all_new_x_test[j*num_new_digits:(j+1)*num_new_digits,...] = mnu.create_new_digits(skeleton_x_test, y_test, 
                                first_digit, second_digit, num_new_digits, 
                                  first_start = first_start, second_start = second_start, 
                                  first_inter_length = first_inter_length, second_inter_length = second_inter_length)
    if(second_digit>=0):
        new_y_test[j*num_new_digits:(j+1)*num_new_digits,] = second_digit + 10*first_digit
    else:
        new_y_test[j*num_new_digits:(j+1)*num_new_digits,] = first_digit

#%%
sample = np.random.choice(len(all_new_x_train),size=1,replace=False)
plt.imshow(all_new_x_train[sample].squeeze(),cmap='turbo')
plt.title('Sample : {0} | Label : {1}'.format( sample, str(new_y_train[sample])))
plt.colorbar()
plt.show()
#%% Histogram
sample = np.random.choice(len(all_new_x_train),size=1,replace=False)

hist, bin_edges = np.histogram(all_new_x_train,bins=20, density=False)

bin_middle = (bin_edges[1:]+bin_edges[:-1])/2

plt.figure()

plt.plot(bin_middle,hist,'oc')
plt.title('Sample : {0} | Label : {1}'.format( sample, str(new_y_train[sample])))
plt.show()

#%% Gaussian filter

def gaussian_filter(u,skeletons,truncate,radius,sigma1, sigma2):
    new_u = np.zeros(u.shape)
    for i in tqdm(range(len(u))):
        sigma = np.random.uniform(sigma1,sigma2)
        new_u[i] = scipy.ndimage.gaussian_filter(skeletons[i],sigma,order=0,
                                                 output=None, mode='constant',
                                                 cval=0.0, truncate=truncate, radius=radius)
    return new_u

all_new_x_train = gaussian_filter(all_new_x_train, all_new_x_train, data['truncate_gaussianFilter'], data['radius_gaussianFilter'],data['sigma1'],data['sigma2'])

all_new_x_test = gaussian_filter(all_new_x_test, all_new_x_test, data['truncate_gaussianFilter'], data['radius_gaussianFilter'],data['sigma1'],data['sigma2'])



#%%
sample = np.random.choice(len(all_new_x_train),size=1,replace=False)
plt.imshow(all_new_x_train[sample].squeeze(),cmap='turbo',vmax=1)
plt.title('Sample : {0} | Label : {1}'.format( sample, str(new_y_train[sample])))
plt.colorbar()
plt.show()

#%%
sample = np.random.choice(len(all_new_x_train),size=1,replace=False)

hist, bin_edges = np.histogram(all_new_x_train[all_new_x_train>0],bins=20, density=False)

bin_middle = (bin_edges[1:]+bin_edges[:-1])/2

plt.figure()

plt.plot(bin_middle,hist,'oc')
plt.title('Sample : {0} | Label : {1}'.format( sample, str(new_y_train[sample])))
plt.show()

#%% Translation

def translation(num_samples,a,b,u):
    #random_indices = np.random.choice(len(u),size=num_samples,replace=False)
    random_indices = random.sample(range(len(u)),num_samples)
    translation_x = np.random.uniform(-a,a,size=num_samples)
    translation_y = np.random.uniform(-b,b,size=num_samples)
    for i in tqdm(range(num_samples)):
        translation = AffineTransform(translation=(translation_x[i],translation_y[i]))
        u[random_indices[i]] = transform.warp(u[random_indices[i]],translation.inverse)
    return u

all_new_x_train  = translation(data['num_train_samples_translation'],data['a_translation'],data['b_translation'],all_new_x_train)
all_new_x_test = translation(data['num_test_samples_translation'],data['a_translation'],data['b_translation'],all_new_x_test)



#%% Shear

def shear(num_samples, shear_x, shear_y, u):
    random_indices = random.sample(range(len(u)),num_samples)
    shear_horizental = np.random.uniform(-shear_x,shear_x,size=num_samples)
    shear_vertical = np.random.uniform(-shear_y,shear_y,size=num_samples)
    for i in tqdm(range(num_samples)):
        shear = AffineTransform(shear=(shear_horizental[i],shear_vertical[i]))
        u[random_indices[i]] = transform.warp(u[random_indices[i]],shear.inverse)
    return u

all_new_x_train = shear(data['num_train_samples_shear'],data['x_shear'],data['y_shear'],all_new_x_train)
all_new_x_test  = shear(data['num_test_samples_shear'],data['x_shear'],data['y_shear'],all_new_x_test)


#%% Rotation

def rotation(num_samples, angle, u):
    random_indices = random.sample(range(len(u)),num_samples)
    angles = np.random.uniform(-angle,angle,size=num_samples)

    for i in tqdm(range(num_samples)):
        rotation = AffineTransform(rotation=np.deg2rad(angles[i]))
        u[random_indices[i]] = transform.warp(u[random_indices[i]],rotation.inverse)
    return u

all_new_x_train = rotation(data['num_train_samples_rotation'],data['angle_rotation'],all_new_x_train)
all_new_x_test = rotation(data['num_test_samples_rotation'],data['angle_rotation'],all_new_x_test)


#%% Lines
def pointOnImageEdges(w,h):
    edge1 = np.random.choice(['top','bottom','left','right'])
    if edge1 == 'top':
        edge2 = np.random.choice(['left','right'])
        if edge2 == 'left':
            return (np.random.randint(0,w),0),(0,np.random.randint(0,h))
        else :
            return (np.random.randint(0,w),0),(w-1,np.random.randint(0,h))
    elif edge1 == 'bottom':
        edge2 = np.random.choice(['left','right'])
        if edge2 == 'left':
            return (np.random.randint(0,w),h-1),(0,np.random.randint(0,h))
        else :
            return (np.random.randint(0,w),h-1),(w-1,np.random.randint(0,h))
    elif edge1 == 'left':
        edge2 = np.random.choice(['top','bottom'])
        if edge2 == 'top':
            return(0,np.random.randint(0,h)),(np.random.randint(0,w),0)
        else :
            return(0,np.random.randint(0,h)),(np.random.randint(0,w),h-1)
    else :
        edge2 = np.random.choice(['top','bottom'])
        if edge2 == 'top':
            return(w-1,np.random.randint(0,h)),(np.random.randint(0,w),0)
        else :
            return(w-1,np.random.randint(0,h)),(np.random.randint(0,w),h-1)
        


def lines(num_samples,w,h,u):
    random_indices = np.random.choice(len(u),size=num_samples,replace=False)
    width_line = np.random.randint(2,3,num_samples)
    intensity = np.random.uniform(0.85,1,num_samples)
    for i in tqdm(range(num_samples)):
        start_point, end_point = pointOnImageEdges(w, h)
        cv2.line(u[random_indices[i]],start_point,end_point,intensity[i],width_line[i])
    return u

all_new_x_train = lines(data['num_train_samples_lines'],54,28,all_new_x_train)
all_new_x_test = lines(data['num_test_samples_lines'],54,28,all_new_x_test)


#%% 
idx = np.random.randint(len(all_new_x_train))
plt.figure()
plt.imshow(all_new_x_train[idx], cmap='gray',vmax=1)
plt.colorbar()
plt.title('Sample : {0} | Label : {1}'.format( idx, str(new_y_train[idx])))
plt.show()
  

#%% Save

with open(os.path.join(input_path, 'new_x_train_withoutnoise_gaussianFilter.pickle'), 'wb') as f:
    pickle.dump(all_new_x_train, f)
    
with open(os.path.join(input_path, 'new_y_train_withoutnoise_gaussianFilter.pickle'), 'wb') as f:
    pickle.dump(new_y_train, f)
    
with open(os.path.join(input_path, 'new_x_test_withoutnoise_gaussianFilter.pickle'), 'wb') as f:
    pickle.dump(all_new_x_test, f)
    
with open(os.path.join(input_path, 'new_y_test_withoutnoise_gaussianFilter.pickle'), 'wb') as f:
    pickle.dump(new_y_test, f)

with open(os.path.join(param_path,'data_withoutnoise_gaussianFilter.json'), 'w') as file:
    json.dump(data, file, indent=4)
