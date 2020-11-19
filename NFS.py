#author: Kang Gu
import numpy as np
import Deconv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import optimizers
from models import resnet, mcnn, mcdcnn, tlenet

#build nfs component
def build_NFS(input_dim, feature_subset=None):
    n_t = input_dim[0]
    n_var = input_dim[1]
    
    if feature_subset is not None:
        index = feature_subset
    else:
        index = [i for i in range(n_var)]
    
    input_layer = keras.layers.Input((n_t, n_var))
    processed_var = []
    splited_var = keras.layers.Lambda(lambda x: Deconv.split(x))(input_layer)
    
    feature_num = 1
    feature_dim = 1
    for i in index:
        single_var = Deconv.DecomposedConv(feature_num, feature_dim)(splited_var[i])
        processed_var.append(single_var)
        
    output = keras.layers.Concatenate(axis=2)(processed_var)
    
    model = keras.models.Model(inputs = input_layer, outputs = output)
    
    return model


#build the whole framework
def build_model(input_dim, downstream_dim, feature_subset=None, downstream="LSTM", enable_fs = False):
    input_layer = keras.layers.Input(input_dim)
    NFS = build_NFS(input_dim, feature_subset)
    downstream = dstmodel(downstream, downstream_dim)
    
    MAP = NFS(input_layer)
    Connection_layer = keras.layers.BatchNormalization(name="feature_selection")
    MAP = Connection_layer(MAP)
    #MAP = keras.layers.Conv1D(2*downstream_dim[1], 1, padding="same", activation="relu",  kernel_regularizer=l1_l2(0.01, 0))(MAP)
    MAP = keras.layers.Conv1D(downstream_dim[1], 1, padding="same", activation="relu",  kernel_regularizer=l1_l2(0.01, 0))(MAP)
    output = downstream(MAP)
    
    model = keras.models.Model(inputs = input_layer, outputs = output)
    OPT = optimizers.Adam(lr=0.001)
    if enable_fs:
        model.compile(loss= Deconv.loss(Connection_layer, 0.001), optimizer=OPT ,metrics=["categorical_accuracy"]) 
        
    else:
        model.compile(loss = "categorical_crossentropy", optimizer=OPT, metrics=["categorical_accuracy"])
        
    return model

#build the downstream model given the choice
def dstmodel(name, dim,output_dim=2):
    name = name.lower()
    if name=="lstm":
       input_layer = keras.layers.Input(dim)
       MAP = keras.layers.LSTM(32, kernel_regularizer=l1_l2(0.01,0.01))(input_layer)
       output = keras.layers.Dense(output_dim, activation="softmax")(MAP)
       dst = keras.models.Model(inputs = input_layer, outputs=output)
       
    elif name=="rnn":
       input_layer = keras.layers.Input(dim)
       MAP = keras.layers.SimpleRNN(32, kernel_regularizer=l1_l2(0.01,0.01))(input_layer)
       output = keras.layers.Dense(output_dim, activation="softmax")(MAP)
       dst = keras.models.Model(inputs = input_layer, outputs=output)
        
    elif name=="gru":
       input_layer = keras.layers.Input(dim)
       MAP = keras.layers.GRU(32, kernel_regularizer=l1_l2(0.01,0.01))(input_layer)
       output = keras.layers.Dense(output_dim, activation="softmax")(MAP)
       dst = keras.models.Model(inputs = input_layer, outputs=output)
    elif name=="cnnlstm":
        filter_num = 5
        input_layer = keras.layers.Input(dim)
        map_1 = keras.layers.Conv1D(filter_num,2, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_2 = keras.layers.Conv1D(filter_num,3, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_3 = keras.layers.Conv1D(filter_num,4, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_4 = keras.layers.Conv1D(filter_num,5, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        MAP = keras.layers.concatenate([map_1,map_2, map_3, map_4], axis=2)
        MAP = keras.layers.LSTM(32, kernel_regularizer=l1_l2(0.01,0.01))(MAP)
        output = keras.layers.Dense(output_dim, activation="softmax")(MAP)
        dst = keras.models.Model(inputs = input_layer, outputs=output)
        
        
    elif name=="cnnrnn":
        filter_num = 5
        input_layer = keras.layers.Input(dim)
        map_1 = keras.layers.Conv1D(filter_num,2, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_2 = keras.layers.Conv1D(filter_num,3, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_3 = keras.layers.Conv1D(filter_num,4, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_4 = keras.layers.Conv1D(filter_num,5, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        MAP = keras.layers.concatenate([map_1,map_2, map_3, map_4], axis=2)
        MAP = keras.layers.SimpleRNN(32, kernel_regularizer=l1_l2(0.01,0.01))(MAP)
        output = keras.layers.Dense(output_dim, activation="softmax")(MAP)
        dst = keras.models.Model(inputs = input_layer, outputs=output)
        
        
    elif name=="cnngru":
        filter_num = 5
        input_layer = keras.layers.Input(dim)
        map_1 = keras.layers.Conv1D(filter_num,2, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_2 = keras.layers.Conv1D(filter_num,3, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_3 = keras.layers.Conv1D(filter_num,4, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        map_4 = keras.layers.Conv1D(filter_num,5, padding="same",activation="relu", kernel_regularizer=l1_l2(0.00, 0.01))(input_layer)
        MAP = keras.layers.concatenate([map_1,map_2, map_3, map_4], axis=2)
        MAP = keras.layers.GRU(32, kernel_regularizer=l1_l2(0.01,0.01))(MAP)
        output = keras.layers.Dense(output_dim, activation="softmax")(MAP)
        dst = keras.models.Model(inputs = input_layer, outputs=output)
        
        
    elif name=="resnet":
        downstream = resnet.Classifier_RESNET("", dim, output_dim)  
        dst = downstream.model
    elif name=="mcnn":
        downstream = mcnn.Classifier_MCNN("", [dim], output_dim)
        dst = downstream.model
    elif name =="mcdcnn":
        input_layer = keras.layers.Input(dim)
        splited_fields = keras.layers.Lambda( lambda x:  Deconv.split(x))(input_layer)
        MCDCNN = mcdcnn.Classifier_MCDCNN("", dim, output_dim)
        downstream = MCDCNN.model
        output = downstream(splited_fields)
        dst = keras.models.Model(inputs = input_layer, outputs = output)
        
    else:
        downstream = tlenet.Classifier_TLENET("", dim, output_dim)
        dst = downstream.model
    
    return dst

def feature_selection(model, k):
    fs_layer = model.get_layer("feature_selection")
    scores = fs_layer.get_weights()[0]
    subset = np.argsort(scores)[-k:]  
    subset.sort()  
    
    return subset


def Agnos(Input_dim):
    print("build auto encoder for feature selection")
    
    input_layer = keras.layers.Input(shape=(Input_dim[0], Input_dim[1]))
    encoder = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu', kernel_regularizer = l1_l2(0.01, 0)))(input_layer)
    encoder = keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu"))(encoder)
        
    decoder = keras.layers.TimeDistributed(keras.layers.Dense(32, activation='relu'))(encoder)
    decoder = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(decoder)
    out = keras.layers.TimeDistributed(keras.layers.Dense(Input_dim[1]))(decoder)
    autoencoder = keras.models.Model(inputs=input_layer, outputs=out)
    autoencoder.compile(optimizer='adam', loss='logcosh')
    
    return autoencoder