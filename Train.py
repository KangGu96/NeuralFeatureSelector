# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:35:36 2020

@author: Kang Gu
"""
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from data_loader import load_MTS
from NFS import build_model, feature_selection, dstmodel, Agnos
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def run_NFS(dataset,K=32, perform_fs=True, dst_dim=32, dst_model="LSTM", selected_feature=None):
    X, Y, x_test, y_test = load_MTS(dataset)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    
    model = build_model([x_train.shape[1], x_train.shape[2]], [x_train.shape[1], dst_dim],downstream=dst_model, feature_subset=selected_feature, enable_fs=perform_fs)
    print("train stage I...")
    model.fit( x_train, y_train,
                    epochs=100,
                    batch_size=128, # Default is 32, we can also change it smaller
                    validation_data=(x_val, y_val),
                    verbose=1,
                    shuffle=True,
                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=0.1, 
                                   patience=1, 
                                   verbose=0, 
                                   mode='min', 
                                   baseline=None, 
                                   restore_best_weights=True)]
                    )
    
    score = model.evaluate(x_test,y_test, verbose=0)
    
    
    if perform_fs:
        print("train stage II...")
        selected_feature = feature_selection(model, K)
        new_model =  build_model([x_train.shape[1], x_train.shape[2]], [x_train.shape[1], dst_dim],downstream=dst_model, feature_subset=selected_feature)
        new_model.fit( x_train, y_train,
                    epochs=100,
                    batch_size=128, # Default is 32, we can also change it smaller
                    validation_data=(x_val, y_val),
                    verbose=1,
                    shuffle=True,
                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=0.1, 
                                   patience=3, 
                                   verbose=0, 
                                   mode='min', 
                                   baseline=None, 
                                   restore_best_weights=True)]
                    )
        newscore = new_model.evaluate(x_test, y_test, verbose=0)
        
        
        print('Test accuracy:', score[1], newscore[1])
    else:
        
        print('Test accuracy:', score[1])
    return

def run_baseline(dataset,  dst_model="LSTM"):
    name = dst_model.lower()
    X, Y, x_test, y_test = load_MTS(dataset)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    #y_test = to_categorical(y_test)
   
    model = dstmodel(name, [x_train.shape[1], x_train.shape[2]])
    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    model.fit( x_train, y_train,
                    epochs=100,
                    batch_size=128, # Default is 32, we can also change it smaller
                    validation_data=(x_val, y_val),
                    verbose=1,
                    shuffle=True,
                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=0, 
                                   patience=2, 
                                   verbose=0, 
                                   mode='min', 
                                   baseline=None, 
                                   restore_best_weights=True)])
    
    
    #score = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    accuracy =  accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy )
    
    return

def run_autoencoder(dataset, k):
    X, Y, x_test, y_test = load_MTS(dataset)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    model = Agnos([x_train.shape[1], x_train.shape[2]])
    model.fit( x_train, x_train,
                    epochs=100,
                    batch_size=128, # Default is 32, we can also change it smaller
                    validation_data=(x_val, x_val),
                    verbose=1,
                    shuffle=True,
                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=0, 
                                   patience=2, 
                                   verbose=0, 
                                   mode='min', 
                                   baseline=None, 
                                   restore_best_weights=True)])
    weights = model.layers[1].get_weights()[0]
    score = np.amax(weights, axis=1)
    indices = np.argsort(score)[-k:]  
    indices.sort()
    print(indices)
    return indices

if __name__ =="__main__":
    #Downstream models include "CNNRNN", "CNNGRU", "CNNLSTM", "RESNET", "MCNN","MCDCNN" and "TLENET"
    
    dataset = "FaceDetection"
    perform_fs = True
    dst_dim = 32
    k = 60 ##parameter for feature selection
    downstream_model = "TLENET"
    autoencoder_feature = [0,1,2,3,4,5,7,10,11,13,17,19,20,21,26,27,28,30,31,33,39,40,41,42,
    48,49,50,54,55,56,57,58]

    
    #run_baseline(dataset, dst_model=downstream_model)
    run_NFS(dataset,K=k, perform_fs=perform_fs,dst_model="MCDCNN")
    #run_autoencoder(dataset, dst_dim)
    #run_NFS(dataset, False, dst_model="MCDCNN",selected_feature=autoencoder_feature)