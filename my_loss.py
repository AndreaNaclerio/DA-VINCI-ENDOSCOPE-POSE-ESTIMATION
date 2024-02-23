#IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)
from tensorflow.keras.layers import Concatenate

from keras import backend as K
def my_loss_function(alpha_pos,alpha_orient):
    def loss(y_true, y_pred): #called each iteration (step)

        pred_pos, pred_orient = y_pred[:, :3], y_pred[:, 3:]
        target_pos, target_orient = y_true[:, :3], y_true[:, 3:]

        loss_pos = K.sum(K.abs(pred_pos - target_pos))
        loss_orient = K.sum(K.abs(pred_orient - target_orient))

        total_loss = alpha_pos * loss_pos + alpha_orient * loss_orient
       
        return total_loss
    
    return loss 



# IMPORT DATASET
y = np.load('target.npy')
resized_final = np.load('resized_final.npy')
resized_start = np.load('resized_start.npy')
seed = 42

# SPLIT TRAINING AND VALIDATION
X_train_1 = resized_start[0:1100,:,:,:]
X_val_1 = resized_start[1100:,:,:,:]
X_train_2 = resized_final[0:1100,:,:,:]
X_val_2 = resized_final[1100:,:,:,:]
Y_train = y[0:1100]
Y_val = y[1100:]

X_train_1 =  np.array(X_train_1/255)
X_val_1 = np.array(X_val_1/255)
X_train_2 = np.array(X_train_2/255)
X_val_2 = np.array(X_val_2/255)
Y_train_ar = np.array(Y_train)
Y_val_ar = np.array(Y_val)

print(X_train_1.shape)
print(Y_train.shape)
print(X_val_1.shape)
print(Y_val.shape)





# VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

model =tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(512, 640, 3),
    pooling=None,
    classes=1000,
    classifier_activation=None
)
model.trainable = False

def build_model(input_shape_1, input_shape_2):
    tf.random.set_seed(seed)

    inputs_1 = tfk.Input(shape=input_shape_1, name='Input_1')
    inputs_2 = tfk.Input(shape=input_shape_1, name='Input_2')
    
    x1 = model(inputs_1)
    x2 = model(inputs_2)

    #TODo: per ridurre il numero di parametri possiamo aggiungere un globalAveragePooling layer or un global max pooling
    x1 = tfkl.GlobalMaxPooling2D()(x1)
    x2 = tfkl.GlobalMaxPooling2D()(x2)

    #x1 = tfkl.Flatten(name='Flattening1')(x1)
    #x2 = tfkl.Flatten(name='Flattening2')(x2)

    x = tfkl.Concatenate(name='Concatenation')([x1, x2])

    x = tfkl.Dense(64, activation='relu',kernel_initializer = tfk.initializers.HeUniform(seed))(x)
    x = tfkl.Dropout(0.5)(x)
    x = tfkl.Dense(32, activation='relu',kernel_initializer = tfk.initializers.HeUniform(seed))(x)
    x = tfkl.Dropout(0.5)(x)

    outputs = tfkl.Dense(6,activation='linear')(x)

    tl_model = tfk.Model(inputs=[inputs_1,inputs_2], outputs=outputs, name='model')
    tl_model.compile(loss=my_loss_function(1,0.1), optimizer=tfk.optimizers.Adam(0.001), metrics=['mae'])
  
    return tl_model


# SAVE IMAGE AND MODEL
name_path = 'model_15_my_loss'

import os
folder_path = "/home/anaclerio/" + name_path
try:
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
except FileExistsError:
    print(f"Folder '{folder_path}' already exists.")


def myprint(s):
    with open(folder_path+ '/modelsummary.txt','a') as f:
        print(s, file=f)

model = build_model(X_train_1.shape[1:],X_train_2.shape[1:])
model.summary(print_fn=myprint)

print(np.max(X_train_1))
print(np.max(X_val_1))

# TRAIN MODEL
history = model.fit(
    x = [preprocess_input(X_train_1*255),preprocess_input(X_train_2*255)], #if we look at the preprocess of VGG we can see that the input should have values between 0-255
    y = Y_train_ar,
    batch_size = 6,
    epochs = 20,
    validation_data = ([preprocess_input(X_val_1*255),preprocess_input(X_val_2*255)], Y_val_ar),
    callbacks = [tfk.callbacks.EarlyStopping(monitor='val_mae', mode='min', patience=12, restore_best_weights=True),
                 tfk.callbacks.ReduceLROnPlateau(monitor='val_my_loss_function',patience=6,factor=0.9,mode='min', min_lr=1e-6)]
).history


#import pickle
#with open(folder_path+'history', 'wb') as file_pi:
#    pickle.dump(history.history, file_pi)
#

model.save(folder_path + '/model', )



from PIL import Image
# Find the epoch with the highest validation accuracy
best_epoch = np.argmax(history['val_loss'])

# Plot training and validation performance metrics
plt.figure(figsize=(20, 5))

# Plot training and validation loss
plt.semilogy(history['loss'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=3)
plt.semilogy(history['val_loss'], label='Validation', alpha=0.8, color='#4D61E2', linewidth=3) #come faccio ad inserirla nella loss che ho definito come funzione ??????
plt.legend(loc='upper left')
plt.title('my_loss_function')
plt.grid(alpha=0.3)
plt.savefig(folder_path +'/plot1.png')

plt.figure(figsize=(20, 5))

# Plot training and validation accuracy, highlighting the best epoch
plt.semilogy(history['mae'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=3)
plt.semilogy(history['val_mae'], label='Validation', alpha=0.8, color='#4D61E2', linewidth=3)
plt.legend(loc='upper left')
plt.title('MEA')
plt.grid(alpha=0.3)
plt.savefig(folder_path +'/plot2.png')

plt.show()


