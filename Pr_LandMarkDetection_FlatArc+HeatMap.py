# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:44:34 2018

@author: Moha-Thinkpad
"""
from glob import glob

from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import tensorflow.keras
from tensorflow.keras import regularizers
from scipy import io
import argparse


from tensorflow.keras import backend as K
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))

####################################
########################################################################
####################################

def custom_loss_reg (y_true, y_pred):
    #A = tensorflow.keras.losses.mean_squared_error(y_true, y_pred)
    B = tensorflow.keras.losses.mean_absolute_error(y_true, y_pred)
   
    return(B)



import tensorflow as tf



def PreProcess(InputImages):
    
    #output=np.zeros(InputImages.shape,dtype=np.float)
    InputImages=InputImages.astype(np.float)
    for i in range(InputImages.shape[0]):
        try:
            if np.max(InputImages[i,:,:,:])==0:
                print(np.max(InputImages[i,:,:,:]))
                plt.imshow(InputImages[i,:,:,:])
                
            InputImages[i,:,:,:]=InputImages[i,:,:,:]/np.max(InputImages[i,:,:,:])
#            output[i,:,:,:] = (output[i,:,:,:]* 2)-1
        except:
            InputImages[i,:,:]=InputImages[i,:,:]/np.max(InputImages[i,:,:])
#            output[i,:,:] = (output[i,:,:]* 2) -1
            
    return InputImages

####################################
########################################################################
####################################

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "export"])
parser.add_argument("--input_dir",  help="path to folder containing images")
parser.add_argument("--target_dir",  help="where to")
parser.add_argument("--checkpoint",  help="where to ")
parser.add_argument("--output_dir",  help="where to p")
parser.add_argument("--landmarks",  help=" -,-,-")
parser.add_argument("--lr",  help="adam learning rate")

# export options
a = parser.parse_args()


layer_trainable=False

a.max_epochs=1
a.batch_size=10
a.lr=0.0001
a.beta1=0.5

# a.mode="train"
# a.input_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_train_png/'
# a.target_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_train_lm/'
# a.checkpoint='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
# a.output_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
# a.landmarks='43,43,43,43,43'

# a.input_dir='C:\\Users\\User\\Desktop\\New folder\\png'
# a.target_dir='C:\\Users\\User\\Desktop\\New folder\\lm'



#a.mode="test"
#a.batch_size=1
#a.input_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_test_png/'
#a.target_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_test_lm/'
#a.checkpoint='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
#a.output_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
#a.landmarks='43,43,43,43,43'


######## ------------ Config 


# 33,23,16 - 29,15, - 30,20,26 - 5,18,21 - 44,17,41 - 28,22,34, - 27,43,37

#Ind_impo_landmarks_matlab=np.array([5, 6, 15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,41])
#Ind_impo_landmarks_python=Ind_impo_landmarks_matlab-1
#Num_landmarks=25

#Ind_impo_landmarks_matlab=np.array([43,30,21,41,33])
#Ind_impo_landmarks_python=Ind_impo_landmarks_matlab-1
#Num_landmarks=5

StrLandmarks=a.landmarks
StrLandmarks=StrLandmarks.split(",")
Ind_impo_landmarks_matlab=np.array([0,0,0,0,0])
Ind_impo_landmarks_matlab[0]=int(StrLandmarks[0])
Ind_impo_landmarks_matlab[1]=int(StrLandmarks[1])
Ind_impo_landmarks_matlab[2]=int(StrLandmarks[2])
Ind_impo_landmarks_matlab[3]=int(StrLandmarks[3])
Ind_impo_landmarks_matlab[4]=int(StrLandmarks[4])
Ind_impo_landmarks_python=Ind_impo_landmarks_matlab-1
Num_landmarks=5


#Ind_impo_landmarks_matlab=np.array([18,23,37])
#Ind_impo_landmarks_python=Ind_impo_landmarks_matlab-1
#Num_landmarks=3

#Num_landmarks=44

print('============================')
print('============================')
print(datetime.datetime.now())
print('============================')
print('============================')


#########----------------------DATA

from os import listdir
ImageFileNames=[]
FileNames=listdir(a.input_dir)
for names in FileNames:
    if names.endswith(".png"):
        ImageFileNames.append(names)
#LMFileNames=listdir(a.target_dir)
from skimage import io as ioSK
from numpy import genfromtxt

Images=np.zeros((len(ImageFileNames),256,256,3),dtype=np.uint8)    
LandmarkLocations=np.zeros((len(ImageFileNames),2,44),dtype=np.uint8)

for i in range(len(ImageFileNames)):
    Image = ioSK.imread(a.input_dir+'/'+ImageFileNames[i])
    Images[i,:,:,:]=Image    
    FileName=ImageFileNames[i]
    FileName=FileName[:-4]
    
    Landmarks0 = genfromtxt(a.target_dir+'/'+FileName+'.csv', delimiter=',')    
    Landmarks0 = Landmarks0.astype(int)    
    LandmarkLocations[i,0,:]=Landmarks0[:,0]
    LandmarkLocations[i,1,:]=Landmarks0[:,1]
    
    #Landmarks = np.flip(Landmarks0, axis=1)
        
    
#plt.figure()
#plt.imshow(Images[100,:,:,:])
#plt.scatter(LandmarkLocations[100,0,:],LandmarkLocations[100,1,:])


#Ind_impo_landmarks_python=np.arange(Num_landmarks)
    
import gc
gc.collect()

LandmarkLocations_row=LandmarkLocations[:,0,:]
LandmarkLocations_col=LandmarkLocations[:,1,:]
LandmarkLocations_row=LandmarkLocations_row[:,Ind_impo_landmarks_python]
LandmarkLocations_col=LandmarkLocations_col[:,Ind_impo_landmarks_python]



X_train = PreProcess(Images) 
del Images
gc.collect()

from scipy.ndimage import gaussian_filter

Images_HeatMaps=np.zeros((X_train.shape[0],X_train.shape[1],X_train.shape[2],Num_landmarks),dtype=np.float)

Image_heatmap=np.zeros((256,256),dtype=np.float)
for i in range(X_train.shape[0]):
  for k in range(Num_landmarks):
      
#        h=np.argwhere(Images_seg[i,:,:]==2*Ind_impo_landmarks_matlab[k])    
      lms_1=LandmarkLocations_row[i,k]
      lms_2=LandmarkLocations_col[i,k]
      Image_heatmap[:,:]=0
      Image_heatmap[lms_2,lms_1]=1
      Image_heatmap=gaussian_filter(Image_heatmap, sigma=10)
      Image_heatmap=(Image_heatmap/np.max(Image_heatmap))
      Images_HeatMaps[i,:,:,k]=Image_heatmap
        
#plt.figure()
#plt.imshow(Images_HeatMaps[0,:,:,:3])
#plt.figure()
#plt.imshow(X_train[0,:,:,:])
#plt.imshow(Images_HeatMaps[0,:,:,:3], alpha=0.6)


Y_train_heatmap = PreProcess(Images_HeatMaps) 
del Images_HeatMaps
gc.collect()



import os
if not os.path.exists(a.checkpoint):
    os.makedirs(a.checkpoint)
    
if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

if a.mode=='test':
    
    checkpoint_model_file=a.checkpoint+'LandMarkModel'

    from tensorflow.keras.models import load_model
    

    print('loading model ...')
    model_final=load_model(checkpoint_model_file+'_weights.h5', custom_objects={'custom_loss_reg': custom_loss_reg,
                                                                                'tf': tf})    
    
    
    

    print('model is loaded ')
    Images=np.zeros((len(ImageFileNames),256,256,3),dtype=np.float)            
    newLandmarks=np.zeros((Num_landmarks,2),dtype=np.float16)
    
    Y_test_heatmap=Y_train_heatmap
    X_test=X_train
        
    #    fig = plt.figure()
    #    plt.imshow(X_train[0,:,:,:],cmap='gray', alpha=0.95)
    #    plt.imshow(Y_train_heatmap[0,:,:,:],cmap='jet', alpha=0.5)
    #    plt.grid(True)
        
    pred_example_heatmaps=model_final.predict(X_test[:,:,:,:])  
    print('writing results ...')
    for i in range(len(ImageFileNames)):
        # print(i)
        FileName=ImageFileNames[i]
        FileName=FileName[:-4]        

        lms_pred_all=np.zeros((Num_landmarks,2),dtype=np.int)
        lms_True_all=np.zeros((Num_landmarks,2),dtype=np.int)
        for k in range(Num_landmarks):
    #        plt.figure()
    #        plt.imshow(example_segmentation[0,:,:,i], cmap='gray')
    #        plt.imshow(Y_train_heatmap[0,:,:,:],cmap='jet', alpha=0.5)
    #        plt.show()
    
           
            True_chan=np.squeeze(Y_test_heatmap[i,:,:,k])
            lms_True=np.unravel_index(np.argmax(True_chan, axis=None), True_chan.shape)
            lms_True_all[k,:]=lms_True
            
            Pred_chan=np.squeeze(pred_example_heatmaps[i,:,:,k])
            lms_pred=np.unravel_index(np.argmax(Pred_chan, axis=None), Pred_chan.shape)
            lms_pred_all[k,:]=lms_pred
            
            
#            fig, ax = plt.subplots(1, 2)
#            ax[0].imshow(Y_test_heatmap[i,:,:,i])        
#            ax[1].imshow(pred_example_heatmaps[i,:,:,i])
#            plt.show()
    
        np.savetxt(a.output_dir+FileName+'_pred.csv', 
           lms_pred_all , delimiter=",", fmt='%i')

        np.savetxt(a.output_dir+FileName+'_true.csv', 
           lms_True_all , delimiter=",", fmt='%i')
    
        fig = plt.figure()
        plt.imshow(X_test[i,:,:,:],cmap='jet', alpha=0.9)
        plt.scatter(lms_True_all[:,1],lms_True_all[:,0], marker='+', color='red')
        plt.scatter(lms_pred_all[:,1],lms_pred_all[:,0], marker='x', color='blue')
#        plt.grid(True)
        fig.savefig(a.output_dir+FileName+'.png')
        plt.close(fig)  
        



if a.mode=='train':
    


    Input_shape=(X_train.shape[1], X_train.shape[2],X_train.shape[3])



    try: # continue training
        checkpoint_model_file=a.checkpoint+'LandMarkModel'

        from tensorflow.keras.models import load_model

        print('======== loading model ...')
        model_final=load_model(checkpoint_model_file+'_weights.h5', custom_objects={'custom_loss_reg': custom_loss_reg,
                                                                                'tf': tf})  
        print('======== continue training ...')
    except:  # new training
        
        seed = 1    
        import random    
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print('======== new training ...')
        checkpoint_model_file=a.output_dir+'LandMarkModel'
            
        ########### network  
        ngf=32
        kernelSize_1=(9,9)
        InputLayer=tensorflow.keras.layers.Input(shape=(256,256,3))
        x_1=tensorflow.keras.layers.Conv2D(ngf, kernel_size=kernelSize_1, dilation_rate=(1, 1), activation='relu',padding='same',)(InputLayer)
        x_1=tensorflow.keras.layers.Conv2D(2*ngf, kernel_size=kernelSize_1, dilation_rate=(1, 1), activation='relu',padding='same',)(x_1)

        
        kernelSize_2=(9,9)
        x_2=tensorflow.keras.layers.Conv2D(ngf, kernel_size=kernelSize_2, dilation_rate=(2, 2), activation='relu',padding='same',)(InputLayer)
        x_2=tensorflow.keras.layers.Conv2D(2*ngf, kernel_size=kernelSize_2, dilation_rate=(2, 2), activation='relu',padding='same',)(x_2)

        
        
        kernelSize_3=(9,9)
        x_3=tensorflow.keras.layers.Conv2D(ngf, kernel_size=kernelSize_3, dilation_rate=(3, 3), activation='relu',padding='same',)(InputLayer)
        x_3=tensorflow.keras.layers.Conv2D(2*ngf, kernel_size=kernelSize_3, dilation_rate=(3, 3), activation='relu',padding='same',)(x_3)

        
        kernelSize_4=(9,9)
        x_4=tensorflow.keras.layers.Conv2D(ngf, kernel_size=kernelSize_4, dilation_rate=(4, 4), activation='relu',padding='same',)(InputLayer)
        x_4=tensorflow.keras.layers.Conv2D(2*ngf, kernel_size=kernelSize_4, dilation_rate=(4, 4), activation='relu',padding='same',)(x_4)

        
        kernelSize_5=(9,9)
        x_5=tensorflow.keras.layers.Conv2D(ngf, kernel_size=kernelSize_5, dilation_rate=(5, 5), activation='relu',padding='same',)(InputLayer)
        x_5=tensorflow.keras.layers.Conv2D(2*ngf, kernel_size=kernelSize_5, dilation_rate=(5, 5), activation='relu',padding='same',)(x_5)



        
        x_c=tensorflow.keras.layers.concatenate([x_1,x_2,x_3,x_4,x_5],axis=-1)
        x_c=tensorflow.keras.layers.Conv2D(8*ngf, kernel_size=(5,5), dilation_rate=(1, 1), activation='relu',padding='same',)(x_c)
        x_c=tensorflow.keras.layers.Conv2D(4*ngf, kernel_size=(1,1), dilation_rate=(1, 1), activation='relu',padding='same',)(x_c)
        x_c=tensorflow.keras.layers.Conv2D(2*ngf, kernel_size=(1,1), dilation_rate=(1, 1), activation='relu',padding='same',)(x_c)               
        FinalHeatMaps=tensorflow.keras.layers.Conv2D(Num_landmarks, kernel_size=(1,1), dilation_rate=(1, 1), activation='tanh',padding='same',)(x_c)

        
        model_final=Model(inputs=InputLayer,outputs=FinalHeatMaps)
               
        #model_final.summary()
                        

        ###########Train
        
#        from keras.utils import plot_model
#        plot_model(model_final, to_file='model.pdf',show_shapes=True, show_layer_names=False ) 
    
    print('trainable_count =',int(np.sum([K.count_params(p) for p in set(model_final.trainable_weights)])))
    print('non_trainable_count =', int(np.sum([K.count_params(p) for p in set(model_final.non_trainable_weights)])))   
        
    #### compile the model
    UsedOptimizer=optimizers.Adam(lr=a.lr, beta_1=a.beta1)
    #UsedOptimizer=tensorflow.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model_final.compile(loss=custom_loss_reg, optimizer=UsedOptimizer)    
    History=model_final.fit(X_train, Y_train_heatmap,
            batch_size=a.batch_size, shuffle=True, validation_split=0.05,
        epochs=a.max_epochs,
            verbose=1)
    
    
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.grid()
    plt.savefig(a.output_dir+'History_'+str(a.lr)+'.png')
    plt.close()
    
            
    Dict={'History_loss_train':History.history['loss'],
          'History_loss_val':History.history['val_loss'],}
    pickle.dump( Dict, open(a.output_dir+'History_'+str(a.lr)+'.pkl', "wb" ) )
    
#    plt.imshow(X_train[0,:,:,:])
#    plt.imshow(Y_train_heatmap[0,:,:,:])
    
    pred_example_heatmaps=model_final.predict(X_train[0:1,:,:,:])
    lms_pred_all=np.zeros((Num_landmarks,2),dtype=np.int)
    lms_True_all=np.zeros((Num_landmarks,2),dtype=np.int)
    for i in range(Num_landmarks):
#        plt.figure()
#        plt.imshow(example_segmentation[0,:,:,i], cmap='gray')
#        plt.imshow(Y_train_heatmap[0,:,:,:],cmap='jet', alpha=0.5)
#        plt.show()

       
        True_chan=np.squeeze(Y_train_heatmap[0,:,:,i])
        lms_True=np.unravel_index(np.argmax(True_chan, axis=None), True_chan.shape)
        lms_True_all[i,:]=lms_True
        
        Pred_chan=np.squeeze(pred_example_heatmaps[0,:,:,i])
        lms_pred=np.unravel_index(np.argmax(Pred_chan, axis=None), Pred_chan.shape)
        lms_pred_all[i,:]=lms_pred
        
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(Y_train_heatmap[0,:,:,i])        
        ax[1].imshow(pred_example_heatmaps[0,:,:,i])
        plt.show()

    fig = plt.figure()
    plt.imshow(X_train[0,:,:,:],cmap='gray', alpha=0.9)
#    plt.imshow(Y_train_heatmap[0,:,:,:],cmap='jet', alpha=0.5)
    plt.scatter(lms_True_all[:,1],lms_True_all[:,0], marker='+', color='red')
    plt.scatter(lms_pred_all[:,1],lms_pred_all[:,0], marker='x', color='blue')
    plt.grid(True)
    plt.close

    

    
    print('===========training done=================')
    print('============================')
    print(datetime.datetime.now())
    print('============================')
    print('============================')
        
        

    print('Saving model ...')
    model_final.save(checkpoint_model_file+'_weights.h5')

