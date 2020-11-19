# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:36:14 2020

@author: Kang Gu
"""
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sktime.utils.load_data import load_from_arff_to_dataframe
from imblearn.over_sampling import ADASYN
import numpy as np
import matplotlib
import pandas
import os 
import math
from tensorflow.keras.utils import to_categorical

###load dataset in MTS archive from arff file
def preprocess_MTS(dataset):
    train_x, train_y = load_from_arff_to_dataframe(os.path.join(dataset, dataset+"_TRAIN.arff"))
    test_x, test_y = load_from_arff_to_dataframe(os.path.join(dataset, dataset+"_TEST.arff"))
    
    train_size = train_x.shape[0]
    test_size = test_x.shape[0]
    
    nb_dims = train_x.shape[1]
    length = len(train_x.iloc[0,0].index)
    
    train = np.empty((train_size, nb_dims, length))
    test = np.empty((test_size, nb_dims, length))
    train_labels = np.empty(train_size, dtype=np.int)
    test_labels = np.empty(test_size, dtype=np.int)
    
    for i in range(train_size):
        train_labels[i] = train_y[i]
        for j in range(nb_dims):
            dim = train_x.iloc[i,j].to_numpy()
            train[i,j] = dim
    for i in range(test_size):
        test_labels[i] = test_y[i]
        for j in range(nb_dims):
            dim = test_x.iloc[i,j].to_numpy()
            test[i,j] = dim
            
    # Normalizing dimensions independently
    for j in range(nb_dims):
        # Post-publication note:
        # Using the testing set to normalize might bias the learned network,
        # but with a limited impact on the reported results on few datasets.
        # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
        mean = np.mean(np.concatenate([train[:, j], test[:, j]]))
        var = np.var(np.concatenate([train[:, j], test[:, j]]))
        train[:, j] = (train[:, j] - mean) / math.sqrt(var)
        test[:, j] = (test[:, j] - mean) / math.sqrt(var)
    
    np.savez(dataset+"/"+"processed_data.npy", Train_X = train, Train_Y = train_labels, Test_X = test, Test_Y = test_labels )
    
    return 

def load_MTS(dataset):
    
    data = np.load( dataset +"/" + "processed_data.npy.npz")
    
    train_X = data['Train_X']
    train_y = data["Train_Y"]
    test_X = data["Test_X"]
    test_y = data["Test_Y"]
    
    return train_X, train_y, test_X, test_y


###load Physionet2012 dataset
def load_ICU(dataset):
    
    X = np.load(dataset + 'X.npy')
    Y = np.load(dataset + 'Y.npy')
    
    # Define standardization tools
    scalers_x = {}
    std_X = []
    feature_num = X.shape[2]
   
    for i in range(feature_num):
        scalers_x[i] = StandardScaler()
        std_X.append( scalers_x[i].fit_transform( np.squeeze(X[:,:,i])) )
        
    X = np.stack(std_X, axis = 2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    sm = ADASYN(sampling_strategy=0.4)
    n = X_train.shape[0]
    X_train = np.reshape( X_train ,(n,24*37))
    X_train, y_train = sm.fit_sample(X_train, y_train)
    N = X_train.shape[0]
    X_train = np.reshape(X_train, (N, 24,37))
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, y_train, X_test, y_test




###load favorita dataset
def load_favorita(dataset):
    data = np.load(dataset, allow_pickle = True)
    
    # Define standardization tools
    scalers_x = {}
    scalers_y = {}
    std_x = []
    
    convert_list = [0,1,4,9,10,11,12,13,14]
   
    for i in range(15):
        if i in convert_list:
            scalers_x[i] = StandardScaler()
            std_x.append( scalers_x[i].fit_transform( np.squeeze(data[:,:,i])) )
        else:
            std_x.append( np.squeeze(data[:,:,i]))
        
    scalers_y[0] = StandardScaler()
    scalers_y[0].fit_transform(np.squeeze(data[:,-16:,0]))
    data = np.stack(std_x, axis = 2)
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare(data)
    print("data loading accomplished")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare(data, use_subset=True):
    if use_subset:
        data, _ = train_test_split(data, test_size = 0.5)
    
    Train_data, Val_data = train_test_split(data, test_size=0.25)
    
    X_train = Train_data[:,:334,:]
    y_train = Train_data[:, 334:350,:]
    
    X_test = Train_data[:,16:350,:]
    y_test = Train_data[:, 350:366,:]
    
    X_val = Val_data[:,:334,:]
    y_val = Val_data[:,334:350,:]
    
    return X_train.astype('float32'), y_train.astype('float32'), X_val.astype('float32'), y_val.astype('float32'), X_test.astype('float32'), y_test.astype('float32')