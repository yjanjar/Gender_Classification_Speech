
import os, sys
import numpy as np
import soundfile as sf 
import python_speech_features as feat 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Conv1D,Dense,LeakyReLU,Softmax,Activation,Input,Flatten,MaxPooling1D,Dropout,BatchNormalization,GaussianNoise,Reshape,UpSampling1D
from keras.activations import softmax
from keras.layers.merge import concatenate
from keras.models import Model
from keras import regularizers,optimizers
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras import metrics

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# General predefined variables.
seed = 0
winlen = 0.025
nfft = 512
numcep = 50 

dataSize = 50
seed = 123
dropout_fraction = 0.5



def initialize_model(activation, dataSize,d_f):
	'''Function to initialize the Convolutional Neural Network'''
	model = Sequential()

	model.add(Conv1D(input_shape=(dataSize,1),kernel_size=5, strides=1, filters=10,
	             kernel_regularizer = regularizers.l2(0.005),
	             kernel_initializer = glorot_normal(seed)))
	model.add(LeakyReLU(0.01))
	#model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))


	model.add(Conv1D(kernel_size=2, strides=1, filters=10,
	             kernel_regularizer = regularizers.l2(0.005),
	             kernel_initializer = glorot_normal(seed)))
	model.add(LeakyReLU(0.01))
	model.add(BatchNormalization())
	#model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
	#model.add(Dropout(dropout_fraction))


	model.add(Flatten())


	#Layer1 : Dense Layer
	model.add(Dense(100,kernel_regularizer = regularizers.l2(0.005),kernel_initializer=glorot_normal(seed)))
	model.add(LeakyReLU(0.001))
	model.add(BatchNormalization())
	model.add(Dropout(dropout_fraction))


	#Layer2: Dense Layer
	model.add(Dense(50,kernel_regularizer = regularizers.l2(0.005),kernel_initializer=glorot_normal(seed)))
	model.add(LeakyReLU(0.001))
	model.add(Dropout(dropout_fraction))

	#Layer3 : Dense Layer + Output Layer
	model.add(Dense(2,kernel_initializer=glorot_normal(seed),activation=activation))       
	return model


def compile_model(model,learning_rate,loss_function,metrics):
    #optimizer
    optimizer = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-8)

    #Compile the model
    model.compile(optimizer=optimizer,loss=loss_function,metrics=metrics)
    return model



def dataset_from_readers(readers):

	''' Creates dataset from the list of readers.

		Inputs : readers (ID,Gender)
		Outputs: features and associated Gender. '''

	dataset = np.ndarray(shape=(0,numcep))
	G = np.ndarray(shape=(0,1))
	for reader_ID, gender in readers:
	    path_to_reader = os.path.join('../Downloads/LibriSpeech/dev-clean/',str(int(reader_ID)))

	    for source, folder, files in os.walk(path_to_reader):
	        for flac_name in files:
	            if flac_name.endswith(".flac"):
	                filepath = os.path.join(source,flac_name)
	                mean_features = extract_from_file(filepath)
	                dataset = np.append(dataset,mean_features,axis=0)
	                G = np.append(G,np.expand_dims(gender*np.ones(mean_features.shape[0]),axis=1),axis=0)
	                                    
	return dataset, G




def get_mean_MFCC(signal,sr):

	''' Performs MFCC features extraction from a signal.

		Inputs : signal, sampling rate
		Outputs: mfcc features, mean over the entire .flac file of the mfcc features. '''

	features = feat.mfcc(signal,sr,winlen=winlen,winstep=0.01,numcep=numcep,
                     nfilt=numcep,nfft=nfft,appendEnergy=False)
	mean = np.expand_dims(np.mean(features,axis=0),axis=1).T
	   
	return features, mean



def extract_from_file(filepath):

	''' Performs mean MFCC features extraction from .flac file by calling get_mean_MFCC.

		Inputs : filepath 
		Outputs: mean over the entire .flac file contained in filepath. '''

	with open(filepath, 'rb') as f:
                    
	    signal, sr = sf.read(f)
	    features, mean_features =  get_mean_MFCC(signal,sr)

	return mean_features


def train_test(readers, split_ratio):

	''' Performs a train/test split on the data.

		 Inputs : readers, the split ratio desired (common: 80/20 split).
		 Ouputs : IDs for the train and test data. '''

	np.random.seed(seed)

	IDs = np.random.permutation(readers.shape[0])
	split = int(readers.shape[0]*split_ratio)
	train_IDs , test_IDs = IDs[:split] , IDs[split:]

	return train_IDs,test_IDs




def normalize_data(data):

    '''Normalize data: (mean=0,var=1)'''

    means = np.mean(data,axis=0)
    stds = np.std(data,axis=0)
    
    norm_data = (data-means)/stds
   
    
    return norm_data


def visu_results(model,test_data,test_G):

    ''' Prints the Confution Matrix and Classification Report'''

    pred = model.predict(test_data)
    print(" ")
    print("                       Classification report.\n")
    print(classification_report(test_G,pred))

    cfm = confusion_matrix(test_G,pred)
    cfm = cfm.astype('int') #/ cfm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cfm, range(2),range(2))
    plt.figure(figsize = (7,4))
    plt.xlabel('Predicted') 
    plt.ylabel('True Label')
    sn.set(font_scale=1.2)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
    plt.xlabel('Predicted') 
    plt.ylabel('True Label')



def get_one_hot(targets, nb_classes):

    ''' Returns Hot one encoding of the labels according to the number of classes.'''

    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])
