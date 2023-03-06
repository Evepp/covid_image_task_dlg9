# Imports
import warnings
warnings.simplefilter(action='ignore')
import keras
import json
import numpy as np
import glob
import pandas as pd

import urllib.request
from PIL import Image
from keras import layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skimage import io, color
import random
from keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping

######################################################################################################################################
######################################################################################################################################
# SAMPLE RESTRICTION RELATED FUNCTIONS
######################################################################################################################################
######################################################################################################################################

def collect_labels(data, labels):
    
    img_label_dict = {}
    
    # relate image and label index in dictionary
    for i in range(len(labels)):
        img_label_dict[i] = labels[i]
        
        
    # Store all the indexes by label
    BP_index = {k for k,v in img_label_dict.items() if v == 'Bacterial Pneumonia'}
    VP_index = {k for k,v in img_label_dict.items() if v == 'Viral Pneumonia'}
    NP_index = {k for k,v in img_label_dict.items() if v == 'No Pneumonia (healthy)'}
    
    CV_index = {k for k,v in img_label_dict.items() if v == 'COVID-19'}

    return [BP_index, NP_index, VP_index, CV_index], img_label_dict


def restrict_sample(proportion, data, labels):

    classified_instances, img_label_dict = collect_labels(data, labels)

    restricted_sample_indices = set()
    restricted_img_label_dict = {}

    for i in classified_instances[:-1]:
        tmp = random.sample(i, k=int(len(i)*proportion))
        for e in tmp:
            restricted_sample_indices.add(e)

    restricted_sample_indices = restricted_sample_indices.union(classified_instances[-1])
    restricted_sample_labels = [img_label_dict.get(i) for i in restricted_sample_indices]
    restricted_sample = [data[i] for i in restricted_sample_indices]

    return restricted_sample, restricted_sample_labels


def restrict_sample_vp_imb(proportion, data, labels):

    classified_instances, img_label_dict = collect_labels(data, labels)

    restricted_sample_indices = set()
    restricted_img_label_dict = {}

    for i in classified_instances[:-2]:
        tmp = random.sample(i, k=int(len(i)*proportion))
        for e in tmp:
            restricted_sample_indices.add(e)
    
    tmp = random.sample(classified_instances[-2], k=int(len(i)*(proportion+0.2)))
    for e in tmp:
            restricted_sample_indices.add(e)

    restricted_sample_indices = restricted_sample_indices.union(classified_instances[-1])
    restricted_sample_labels = [img_label_dict.get(i) for i in restricted_sample_indices]
    restricted_sample = [data[i] for i in restricted_sample_indices]

    return restricted_sample, restricted_sample_labels


def generate_summary(sum_type, original, new):

    if sum_type == "R":
        sum_type = "RESTRICTED"
    elif sum_type == "A":
        sum_type = "AUGMENTED"

    print(f"SAMPLE {sum_type} SUMMARY")
    print("(Only covering training sample)")
    print("")
    print("")
    print("ORIGINAL TRAINING SAMPLE")
    print(F"Original number of instances: {len(original)}")
    print(f"Original instance distribution by class: \n {pd.Series(original).value_counts()}")
    print("")
    print("RESTRICTED TRAINING SAMPLE")
    print(f"Number of instances in restricted sample: {len(new)}")
    print(f"Instance distribution by class in restricted sample: \n {pd.Series(new).value_counts()}")

######################################################################################################################################
######################################################################################################################################
# DATA AUGMENTATION RELATED FUNCTIONS
######################################################################################################################################
######################################################################################################################################

def rotate90(image):
    return tf.image.rot90(image)

def rotate180(image):
    return tf.image.rot90(rotate90(image))

def rotate270(image):
    return tf.image.rot90(rotate180(image))  

def augment_sample(features, labels, a_rotate = False):

    img_label_dict = {}
    
    # relate image and label index in dictionary
    for i in range(len(labels)):
        img_label_dict[i] = labels[i]
        
        
    # Store all the indexes by label
    BP_index = {k for k,v in img_label_dict.items() if v == 'Bacterial Pneumonia'}
    VP_index = {k for k,v in img_label_dict.items() if v == 'Viral Pneumonia'}
    NP_index = {k for k,v in img_label_dict.items() if v == 'No Pneumonia (healthy)'}
    
    CV_index = {k for k,v in img_label_dict.items() if v == 'COVID-19'}
    
    # Create list with all the previous lists so it can be used to iterate (except covid)
    labels_collect = [BP_index, VP_index, NP_index]
    
    # Select random sample of 50% of each element which is not covid to apply flip transformation
    # The remaining 50% stays the same
    
    not_to_transform = set()
    to_transform = set()
    
    for i in labels_collect:
        for e in i: 
            not_to_transform.add(e)
            
    #Select 15% of no cv instances at random to include augmented copies of these in the dataset
    
    for i in labels_collect:        
        tmp = random.sample(i, k=int(len(i)*0.15))
        for e in tmp:
            to_transform.add(e)
            
    # Creating list for keeping track of the labels for the new data        
    new_labels = []

    for i in not_to_transform:
        new_labels.append(img_label_dict.get(i))
    for i in to_transform:
        new_labels.append(img_label_dict.get(i))
    
    # Transform the sampled and include the sample no_cv images
    new_features_no_cv = []
    
    for i in not_to_transform: # Include regular instances in the new set
            new_features_no_cv.append(features[i])
            
    for i in to_transform: # Include augmented instances are added to the new list
            new_features_no_cv.append(tf.image.flip_left_right(features[i]))
    
    # Include in new feature set covid x-rays + augmented covid x-rays
    new_features_cv = []
    
    for i in CV_index:
        new_features_cv.append(tf.image.flip_left_right(features[i]))
        new_features_cv.append(features[i])
    
    if a_rotate == False:
        
        new_labels_cv = ['COVID-19' for i in range(len(new_features_cv))]
        
        augmentedfeatures = new_features_no_cv + new_features_cv
        augmentedlabels = new_labels + new_labels_cv
        
        return augmentedfeatures, augmentedlabels
        
    # AUGMENTATION - ROTATE    
    elif a_rotate == True:
        
        #Divide no_cv instances in 3 groups (30%, 35%, 35%)
        
        fd_features, features_1, fd_labels, labels_1 = train_test_split(new_features_no_cv, new_labels, 
                                                            test_size=0.3, random_state=42)
        
        features_2, features_3, labels_2, labels_3 = train_test_split(fd_features, fd_labels, 
                                                            test_size=0.5, random_state=42)
        
        # Apply rotate90 to the first group and add both original and rotated to the new feature set alongside the labels
        
        new_features_2 = []
        new_labels_2 = []
        
        for i in range(len(features_1)):
            new_features_2.append(features_1[i])
            new_features_2.append(rotate90(features_1[i]))
            for e in range(2):
                new_labels_2.append(labels_1[i])
        
        # Idem 180
        
        for i in range(len(features_2)):
            new_features_2.append(features_2[i])
            new_features_2.append(rotate180(features_2[i]))
            for e in range(2):
                new_labels_2.append(labels_2[i])
        
        # Idem 270
        
        for i in range(len(features_3)):
            new_features_2.append(features_3[i])
            new_features_2.append(rotate270(features_3[i]))
            for e in range(2):
                new_labels_2.append(labels_3[i])
        
        #Apply all rotations to all cv features and add them to the new set alongside the originals and the labels
        
        for i in range(len(new_features_cv)):
            new_features_2.append(new_features_cv[i])
            new_features_2.append(rotate90(features_3[i]))
            new_features_2.append(rotate180(features_3[i]))
            new_features_2.append(rotate270(features_3[i]))
            for e in range(4):
                new_labels_2.append('COVID-19')
        
        
        augmentedfeatures = new_features_2
        augmentedlabels = new_labels_2
        
        return augmentedfeatures, augmentedlabels

