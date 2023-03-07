# Imports
from PIL import Image

from keras import layers, models
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.regularizers import l2

from skimage import io, color

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU

import glob
import json
import keras
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import random
import tensorflow as tf
import urllib.request

import warnings
warnings.simplefilter(action='ignore')


######################################################################################################################################
######################################################################################################################################
# RANDOM SUBSAMPLING RELATED FUNCTIONS
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
        sum_type = "RESTRICTION"
    elif sum_type == "A":
        sum_type = "AUGMENTATION"

    print(f"SAMPLE {sum_type} SUMMARY")
    print("(Only covering training sample)")
    print("")
    print("")
    print("ORIGINAL TRAINING SAMPLE")
    print(F"Original number of instances: {len(original)}")
    print(f"Original instance distribution by class: \n {pd.Series(original).value_counts()}")
    print("")
    print("NEW TRAINING SAMPLE")
    print(f"Number of instances in new sample: {len(new)}")
    print(f"Instance distribution by class in new sample: \n {pd.Series(new).value_counts()}")

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


######################################################################################################################################
######################################################################################################################################
# BASELINE MODEL
######################################################################################################################################
######################################################################################################################################

def build_baseline_model():
    model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(156, 156, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])

    # Compile the model with appropriate loss function, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



######################################################################################################################################
######################################################################################################################################
# TUNED MODEL
######################################################################################################################################
######################################################################################################################################


# TUNED MODEL 1: 
# BASELINE + LR: Nadam + KFOLD VALIDATION + EARLY STOPPING + LeakyReLU + Extra Dense layer + 3 Dropout layers + Doubled batch size to 64

# FAILED CHANGES VS BASELINE INDICATED WITH A HASHTAG

def build_tuned_model():
    model = keras.Sequential([
    # Convolutional layers
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=LeakyReLU(), input_shape=(156, 156, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=LeakyReLU()),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=LeakyReLU()),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=LeakyReLU()),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=LeakyReLU()),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=LeakyReLU()),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # (FAILED CHANGE VS BASELINE) Adding another pack of Conv2D and MaxPooling2D layers
    #keras.layers.Conv2D(filters=64, kernel_size=(3, 3),  activation=keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.01)),
    #keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  activation=keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.01)),
    #keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation=LeakyReLU()),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation=LeakyReLU()),
    keras.layers.Dropout(0.2),    
    keras.layers.Dense(32, activation=LeakyReLU()),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation='softmax')])
    # Compile the model with appropriate loss function, optimizer, and metrics
    # (FAILED CHANGE VS BASELINE) Adding the optimal using lr schedule for SGD
    optim = keras.optimizers.Nadam(learning_rate=0.001) #(FAILED CHANGE VS BASELINE) optimizer = adam, sgd, rmsprop

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model



######################################################################################################################################
######################################################################################################################################
# RESNET MODEL
######################################################################################################################################
######################################################################################################################################

def build_h_model():
    '''restnet blocks'''
    input_shape = (156, 156, 3)
    inputs = keras.Input(shape=input_shape)
     # Convolutional layers
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(),  padding='same')(inputs)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    #  block 1
    out=x
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.add([out, x])
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    #  block 2
    out=x
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.add([out, x])
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    #  block 3
    out=x
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  activation=keras.layers.LeakyReLU(), padding='same')(x)
    x = keras.layers.add([out, x])
    x = keras.layers.LeakyReLU()(x)
    # Dense layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU())(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU())(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(4, activation='softmax')(x)
    optim = keras.optimizers.Nadam(learning_rate=0.001) 


    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='h_model')

    # Compile the model with appropriate loss function, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model


######################################################################################################################################
######################################################################################################################################
# PLOTTING/PERFORMANCE FUNCTIONS
######################################################################################################################################
######################################################################################################################################

def plot_acc_loss(histo):
     ##Plot for the accuracy of the baseline model 
    accuracy_train = histo.history['accuracy']
    accuracy_val = histo.history['val_accuracy']
    plt.plot(accuracy_train, label='training_accuracy')
    plt.plot(accuracy_val, label='validation_accuracy')
    plt.title('ACCURACY OF THE MODEL')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    ##Plot for the loss of the baseline model 
    loss_train = histo.history['loss']
    loss_val = histo.history['val_loss']
    plt.plot(loss_train, label='training_loss')
    plt.plot(loss_val, label='validation_loss')
    plt.title('LOSS OF MODEL')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return 


def plot_ROC_curve(y_predict,y_test,num_clas): 

    fpr = {}
    tpr = {}
    roc_auc = {}
    #calculating roc for each class
    for i in range(num_clas):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_predict[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
  
  # calculating micro-average ROC curve and  area
    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc_micro = roc_auc_score(y_test.ravel(), y_predict.ravel())

  # Compute macro-average ROC curve and  area
    fpr_macro = np.unique(np.concatenate([fpr[i] for i in range(num_clas)]))
    tpr_macro = np.zeros_like(fpr_macro)
    for i in range(num_clas):
        tpr_macro += np.interp(fpr_macro, fpr[i], tpr[i])
    tpr_macro /= num_clas
    roc_auc_macro = auc(fpr_macro, tpr_macro)

  #Plot the ROC curve for each class using matplotlib.pyplot.plot()
    plt.figure(figsize=(10, 5))
    lw = 2
    for i in range(num_clas):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class %d (area = %0.2f)' % (i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr_micro, tpr_micro,lw=lw, linestyle='--', label='micro-average ROC curve (area = %0.2f)' % (roc_auc_micro))
    plt.plot(fpr_macro, tpr_macro,lw=lw, linestyle='--', label='macro-average ROC curve (area = %0.2f)' % (roc_auc_macro))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of Multiclass')
    plt.legend(loc="lower right")
    plt.show()
    return 


def plot_cm(label_ma,y_predict,y_tes):
    #reversing pred to categorical so to get the labels 
    inverse_label_map = {v: k for k, v in label_ma.items()}  # invert the label_map
    y_pred_decoded_numerical = np.argmax(y_predict, axis=1)
    y_pred_decoded_categorical = np.vectorize(inverse_label_map.get)(y_pred_decoded_numerical)
    #confusion matrix 
    
    cm = confusion_matrix(y_tes, y_pred_decoded_categorical)
    classes = np.unique(y_tes)
    # plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Reds')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, ylabel='True label',
             xlabel='Predicted label')

    # rotate the labels
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    # text annotations like the numbers inside 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.show()
    return


def table_p_r_f1(y_tes,y_predict, label_ma):
  
    inverse_label_map = {v: k for k, v in label_ma.items()}  # invert the label_map
    y_pred_decoded_numerical = np.argmax(y_predict, axis=1)
    y_pred_decoded_categorical = np.vectorize(inverse_label_map.get)(y_pred_decoded_numerical)

    print(classification_report(y_test, y_pred_decoded_categorical))



def plot_comparison_acc_loss(models_compared,name_models, numepoch):
    epoch = range(1, numepoch+1)
    clr=['b','r','y','g','k','c']
    plt.figure(figsize=(10, 8))
    # Plotting the results of validation accuracy and loss for the baseline and tuned models
    plt.subplot(2, 2, 1)
    i=0
    while i<len(models_compared):
      plt.plot(epoch, models_compared[i].history['val_accuracy'], clr[i], label=name_models[i])
      i+=1
    plt.title('Validation accuracy comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    i=0
    plt.subplot(2, 2, 2)
    while i<len(models_compared):
      plt.plot(epoch, models_compared[i].history['val_loss'], clr[i], label=name_models[i])
      i+=1
    plt.title('Validation loss comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    i=0
    # Plotting the results of training accuracy and loss for the baseline and tuned models  
    plt.subplot(2, 2, 3)
    while i<len(models_compared):
      plt.plot(epoch, models_compared[i].history['accuracy'], clr[i], label=name_models[i])
      i+=1
    plt.title('Training accuracy comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    i=0
    plt.subplot(2, 2, 4)
    while i<len(models_compared):
      plt.plot(epoch, models_compared[i].history['loss'], clr[i], label=name_models[i])
      i+=1
    plt.title('Training loss comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplots_adjust(hspace=0.4,wspace=0.4)
    plt.show()
    

######################################################################################################################################
######################################################################################################################################
# GRID SEARCH 
######################################################################################################################################
######################################################################################################################################
    
    
    
def build_optimized_model(alpha1,alpha2,alpha3,alpha4,alpha5):
    model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=alpha1), input_shape=(156, 156, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=alpha1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=alpha2)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=alpha2)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3),activation=keras.layers.LeakyReLU(alpha=alpha3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=alpha3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=keras.layers.LeakyReLU(alpha=alpha4)),
        keras.layers.Dense(32,activation=keras.layers.LeakyReLU(alpha=alpha5)),
        keras.layers.Dense(4,activation='softmax')
    ])

    # Compile the model with appropriate loss function, optimizer, and metrics
    optim = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model
        
        
######################################################################################################################################
######################################################################################################################################
# OPTUNA
######################################################################################################################################
######################################################################################################################################
        
def create_model(trial):
    model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(
            filters=trial.suggest_categorical('filters_1', [32, 64, 128]),
            kernel_size=trial.suggest_categorical('kernel_size_1', [(3, 3), (5, 5)]),
            activation='relu',
            input_shape=(156, 156, 3)
        ),
        keras.layers.Conv2D(
            filters=trial.suggest_categorical('filters_2', [32, 64, 128]),
            kernel_size=trial.suggest_categorical('kernel_size_2', [(3, 3), (5, 5)]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            filters=trial.suggest_categorical('filters_3', [32, 64, 128]),
            kernel_size=trial.suggest_categorical('kernel_size_3', [(3, 3), (5, 5)]),
            activation='relu'
        ),
        keras.layers.Conv2D(
            filters=trial.suggest_categorical('filters_4', [32, 64, 128]),
            kernel_size=trial.suggest_categorical('kernel_size_4', [(3, 3), (5, 5)]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            filters=trial.suggest_categorical('filters_5', [32, 64, 128]),
            kernel_size=trial.suggest_categorical('kernel_size_5', [(3, 3), (5, 5)]),
            activation='relu'
        ),
        keras.layers.Conv2D(
            filters=trial.suggest_categorical('filters_6', [32, 64, 128]),
            kernel_size=trial.suggest_categorical('kernel_size_6', [(3, 3), (5, 5)]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(units=trial.suggest_int('units_1', 16, 128), activation='relu'),
        keras.layers.Dense(units=trial.suggest_int('units_2', 16, 128), activation='relu'),
        keras.layers.Dense(units=4, activation='softmax')
    ])

    # Compile the model with appropriate loss function, optimizer, and metrics
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        ),
        metrics=['accuracy']
    )
    
    return model

        
def objective(trial):
    # Create the model with the hyperparameters suggested by Optuna
    model = create_model(trial)
    
    # Train the model for 5 epochs
    history = model.fit(
        X_train_norm,
        y_train_onehot,
        batch_size=32,
        epochs=5,
        validation_data=(X_val_norm, y_val_onehot),
        verbose=0
    )
        
# Evaluate the model
    loss = history.history['val_loss'][-1]
    return loss